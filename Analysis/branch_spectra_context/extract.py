#!/usr/bin/env python
"""Hook every SPAM block (ExSPAM) of SPANetV2-S18 for one checkpoint and both resolutions; store spectra and maps.

    python extract.py --arch spanetv2_s18_pure --ckpt IN1K=<path> --ckpt COCO/MaskRCNN=<path> ... [--device cpu]

The classification model of image_classification/models/spanetv2.py (input_size 224, so the SRF grids are the stored
224 ones and are resized by ExSPAM.forward itself at 512) receives either the ImageNet state_dict or the 'backbone.'
part of a downstream state_dict (prefix removed). No code is changed; only hooks are registered (ExSPAM.forward,
image_classification/models/spanetv2.py lines 237-269):
  x1_h  local_conv_{h+1}            forward hook, input  (B, C/4, H, W)      = ctx of head h
  x2_h  local_conv_{h+1}            forward hook, output (B, C/4, H, W)
  x3_h  proj                        forward pre-hook, input s_in (B, C, H, W); s_in.view(B, C/4, 4, H, W)[:, :, h]
  x4_h  proj[0] applied (inside the pre-hook, no_grad, via Conv2d._conv_forward so that no hook fires) to s_in with
        every channel of the other heads set to 0;
        proj[0] is Conv2d(C, 2C, 1, groups=C/4, bias=False), so sum_h x4_h = proj[0](s_in); checked against the
        actual proj[0] output (forward hook on proj[0])
  s_ctx proj                        forward hook, output (B, C, H, W)     (= the notebook's vis_ctx before permute)
  q     query                       forward hook, output (B, H, W, C)
  m     proj_out                    forward pre-hook, input (B, H, W, C)  = q * s_ctx
For every tensor X (B, C, H, W): P = mean_c |FFT2(X_c, norm='ortho')|^2 on the full grid. Stored per image:
ring means of P (rings r = round(sqrt(iy^2 + ix^2)), rho = r / (H/2)), ring means of log10(P / P_DC), P_DC, and band
means over grid cells (low 0 < rho <= 0.25, mid 0.25 < rho <= 0.5, high rho > 0.5) of P and of log10(P / P_DC).
Context maps: per block, mean over channels of |s_ctx| (B, H, W).
Writes data/raw/<arch>/<label>_R<res>.npz (block spectra), <label>_R<res>_maps.npz (block maps) and
<label>_load.json.
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
AN = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(AN, 'freq_robustness'))
import common as C  # noqa: E402

RES = (224, 512)
MEAN = torch.tensor(C.IMNET_MEAN).view(1, 3, 1, 1)
STD = torch.tensor(C.IMNET_STD).view(1, 3, 1, 1)


class SpecGrid:
    def __init__(self, H, W, dev):
        iy = torch.fft.fftfreq(H, 1 / H, device=dev).view(-1, 1)
        ix = torch.fft.fftfreq(W, 1 / W, device=dev).view(1, -1)
        r = torch.sqrt(iy ** 2 + ix ** 2)
        self.ring = torch.round(r).long()
        self.R = int(self.ring.max()) + 1
        rho = r / (H / 2)
        self.band = [(rho > 0) & (rho <= 0.25), (rho > 0.25) & (rho <= 0.5), rho > 0.5]
        self.onehot = torch.nn.functional.one_hot(self.ring.flatten(), self.R).double()   # (HW, R)
        self.cnt = self.onehot.sum(0)

    def stats(self, X):
        X = X.double()
        P = (torch.fft.fft2(X, norm='ortho').abs() ** 2).mean(1)          # (B, H, W)
        B = P.shape[0]
        pdc = P[:, 0, 0]
        L = torch.log10(P / pdc.view(-1, 1, 1))
        Pf, Lf = P.reshape(B, -1), L.reshape(B, -1)
        ringP = Pf @ self.onehot / self.cnt
        ringL = Lf @ self.onehot / self.cnt
        bandP = torch.stack([P[:, m].mean(1) for m in self.band], 1)
        bandL = torch.stack([L[:, m].mean(1) for m in self.band], 1)
        return dict(ringP=ringP.float().cpu().numpy(), ringL=ringL.float().cpu().numpy(), PDC=pdc.float().cpu().numpy(),
                    bandP=bandP.float().cpu().numpy(), bandL=bandL.float().cpu().numpy())


def load_ckpt(path):
    info = {}
    try:
        ck = torch.load(path, map_location='cpu', weights_only=True)
        info['weights_only'] = True
    except Exception as e:
        info['weights_only'] = False
        info['weights_only_true_error'] = f'{type(e).__name__}: ' + ' '.join(str(e).split())[:300]
        ck = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(ck, dict) and 'state_dict' in ck:
        sd = {k[len('backbone.'):]: v for k, v in ck['state_dict'].items() if k.startswith('backbone.')}
        info['source'] = "ckpt['state_dict'], keys starting with 'backbone.' (prefix removed)"
        info['n_non_backbone_keys_ignored'] = sum(1 for k in ck['state_dict'] if not k.startswith('backbone.'))
    else:
        sd = ck
        info['source'] = 'flat state_dict'
    return sd, info


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()


def run(net, x, dev):
    """Forward x (B,3,R,R, normalised) with hooks on every SPAM block; return dict of stats and maps."""
    spam = []
    for s, stage in enumerate(net.stages):
        for b, blk in enumerate(stage):
            if type(blk.token_mixer).__name__ == 'ExSPAM':
                spam.append((s + 1, b, blk.token_mixer))
    grids, out, maps, checks, handles = {}, {}, {}, {}, []

    def grid(H, W):
        if (H, W) not in grids:
            grids[(H, W)] = SpecGrid(H, W, dev)
        return grids[(H, W)]

    def put(key, X):
        out[key] = grid(*X.shape[-2:]).stats(X)

    for s, b, tm in spam:
        tag = f's{s}b{b}'
        nh = tm.num_heads
        for h in range(nh):
            def conv_hook(m, i, o, h=h, tag=tag):
                put(f'{tag}|x1_h{h}', i[0])
                put(f'{tag}|x2_h{h}', o)
            handles.append(getattr(tm, f'local_conv_{h + 1}').register_forward_hook(conv_hook))

        def proj_pre(m, i, tag=tag, tm=tm, nh=nh):
            s_in = i[0]
            B, Cc, H, W = s_in.shape
            v = s_in.view(B, Cc // nh, nh, H, W)
            tot = 0
            with torch.no_grad():
                for h in range(nh):
                    put(f'{tag}|x3_h{h}', v[:, :, h])
                    keep = torch.zeros_like(v)
                    keep[:, :, h] = v[:, :, h]
                    p0 = tm.proj[0]               # _conv_forward: same op as proj[0](.), without firing its hooks
                    x4 = p0._conv_forward(keep.view(B, Cc, H, W), p0.weight, p0.bias)
                    put(f'{tag}|x4_h{h}', x4)
                    tot = tot + x4
                put(f'{tag}|sum_x4', tot)
                checks[tag] = dict(sum_x4=tot)

        def proj0_post(m, i, o, tag=tag):
            ref = o.detach()
            d = (checks[tag]['sum_x4'] - ref)
            checks[tag] = dict(max_abs_diff=float(d.abs().max()), rel_l2=float(d.norm() / ref.norm()),
                               bias=m.bias is not None)

        def proj_post(m, i, o, tag=tag):
            put(f'{tag}|s_ctx', o)
            maps[tag] = o.abs().mean(1).float().cpu().numpy()

        def q_post(m, i, o, tag=tag):
            put(f'{tag}|q', o.permute(0, 3, 1, 2))

        def m_pre(m, i, tag=tag):
            put(f'{tag}|m', i[0].permute(0, 3, 1, 2))
        handles += [tm.proj.register_forward_pre_hook(proj_pre), tm.proj[0].register_forward_hook(proj0_post),
                    tm.proj.register_forward_hook(proj_post), tm.query.register_forward_hook(q_post),
                    tm.proj_out.register_forward_pre_hook(m_pre)]
    with torch.no_grad():
        logits = net(x)
    for h_ in handles:
        h_.remove()
    rho = {f'{H}x{W}': (np.arange(g.R) / (H / 2)).astype(np.float32) for (H, W), g in grids.items()}
    return out, maps, checks, rho, logits.argmax(1).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', default='spanetv2_s18_pure')
    ap.add_argument('--ckpt', action='append', required=True, help='LABEL=PATH')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--threads', type=int, default=16)
    ap.add_argument('--bs', type=int, default=13)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    dev = torch.device(args.device)
    raw = os.path.join(HERE, 'data', 'raw', args.arch)
    os.makedirs(raw, exist_ok=True)
    mod = C.load_module('spanetv2')
    for spec in args.ckpt:
        label, path = spec.split('=', 1)
        t0 = time.time()
        net = getattr(mod, args.arch)()
        sd, info = load_ckpt(path)
        res = net.load_state_dict(sd, strict=False)
        info.update(path=os.path.relpath(path, AN), sha256=sha256(path), missing=list(res.missing_keys),
                    unexpected=list(res.unexpected_keys), n_loaded=len(sd) - len(res.unexpected_keys))
        net = net.to(dev).eval()
        safe = label.replace('/', '_')
        for R in RES:
            imgs = np.load(os.path.join(HERE, 'data', f'images_R{R}.npy'))
            acc_out, acc_maps, checks, preds = {}, {}, {}, []
            for i in range(0, len(imgs), args.bs):
                x = torch.from_numpy(imgs[i:i + args.bs]).permute(0, 3, 1, 2).float().div(255)
                x = ((x - MEAN) / STD).to(dev)
                o, mp, ck, rho, pr = run(net, x, dev)
                preds.append(pr)
                for k, v in o.items():
                    for kk, vv in v.items():
                        acc_out.setdefault(f'{k}|{kk}', []).append(vv)
                for k, v in mp.items():
                    acc_maps.setdefault(f'map|{k}', []).append(v)
                for k, v in ck.items():
                    checks.setdefault(k, []).append(v)
            arrs = {k: np.concatenate(v, 0) for k, v in acc_out.items()}
            arrs.update({f'rho|{k}': v for k, v in rho.items()})
            np.savez_compressed(os.path.join(raw, f'{safe}_R{R}.npz'), **arrs)            # block spectra
            np.savez_compressed(os.path.join(raw, f'{safe}_R{R}_maps.npz'),                 # block context maps
                                **{k: np.concatenate(v, 0) for k, v in acc_maps.items()})
            info[f'R{R}'] = dict(sum_x4_check={k: dict(max_abs_diff=max(c['max_abs_diff'] for c in v),
                                                      rel_l2_max=max(c['rel_l2'] for c in v), bias=v[0]['bias'])
                                              for k, v in checks.items()},
                                 imagenet_top1_pred=[int(p) for p in np.concatenate(preds)])
        info['seconds'] = time.time() - t0
        json.dump(info, open(os.path.join(raw, f'{safe}_load.json'), 'w'), indent=1)
        mx = max(v['max_abs_diff'] for R in RES for v in info[f'R{R}']['sum_x4_check'].values())
        print(f'{label}: missing {len(info["missing"])}, unexpected {len(info["unexpected"])}, '
              f'max|sum x4 - proj[0]| {mx:.2e}, {info["seconds"]:.0f}s', flush=True)
        del net


if __name__ == '__main__':
    main()
