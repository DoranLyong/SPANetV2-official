#!/usr/bin/env python
"""E1 / E2 data: forward hooks on the token mixers of SPANetV2-S18-pure, ConvFormer-S18, DFFormer-S18, CAFormer-S18
(ImageNet-1K weights, eval mode, fp32, CPU) for the 12 images of select_images.py, plus the filter parameters.

    python extract.py [--weights-manifest ../freq_robustness/data/weights_manifest.json]

Only hooks are used; no model code is changed. Hook points (block b = last block of stages 1-2, middle block of stage 3):
  SPANetV2   stages.{s}.{b}.token_mixer.proj      forward-pre-hook: its input s_ctx (B, C, H, W) is the stack of the
                                                 four SRF outputs s_i (irfft2 output of ExSPAM.forward, before proj);
                                                 s_ctx.view(B, C/4, 4, H, W)[:, :, i] = head i (kernel 3 + 2i)
             stages.{s}.{b}.token_mixer           forward hook: mixer output (B, H, W, C)
  ConvFormer stages.{s}.{b}.token_mixer.dwconv    forward hook: depthwise-conv output (B, 2C, H, W)
             stages.{s}.{b}.token_mixer           forward hook: mixer output
  DFFormer   stages.{s}.{b}.token_mixer.act2      forward-pre-hook (act2 = nn.Identity): its input is the irfft2 output
                                                 right after the dynamic filter (B, H, W, 2C)
             stages.{s}.{b}.token_mixer.reweight  forward hook: routing logits; routing = softmax over the filter axis
             stages.{s}.{b}.token_mixer           forward hook: mixer output
  CAFormer   stages.2.{b}.token_mixer             forward hook (stage 3 only): attention output
Spectrum of a feature map X (C, H, W): P = mean_c |FFT2(X_c)|^2 on the full grid, L = log10(P / P_DC), averaged over
integer-radius rings r = round(sqrt(ky^2 + kx^2) * H); rho = r / (H / 2) (axis Nyquist = 1).
Maps: channel L2 norm sqrt(sum_c X_c^2) per pixel.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
AN = os.path.dirname(HERE)
FR = os.path.join(AN, 'freq_robustness')
sys.path.insert(0, FR)
import common as C  # noqa: E402  (model builders and weight loading of the frequency-robustness study)

_spec = importlib.util.spec_from_file_location('_srf', os.path.join(AN, 'srf_task_analysis', 'analyze_srf.py'))
SRF = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(SRF)          # Grid: rfft half-grid weights, rings, kernel DTFT (same definitions as C-4)

MODELS = ['spanetv2_s18_pure', 'convformer_s18', 'dfformer_s18', 'caformer_s18']
STAGES = [1, 2, 3]


def block_index(net, s):
    n = len(net.stages[s - 1])
    return n - 1 if s in (1, 2) else n // 2


def spectrum_rings(X):
    """X (N, C, H, W) -> (N, R) ring means of log10(P/P_DC), plus linear P (N, H, W)."""
    N, Cc, H, W = X.shape
    P = (torch.fft.fft2(X.double()).abs() ** 2).mean(1)                     # (N, H, W)
    L = torch.log10(P / P[:, :1, :1])
    iy = torch.fft.fftfreq(H, 1 / H).view(-1, 1)
    ix = torch.fft.fftfreq(W, 1 / W).view(1, -1)
    ring = torch.round(torch.sqrt(iy ** 2 + ix ** 2)).long()
    R = int(ring.max()) + 1
    out = torch.stack([L[:, ring == r].mean(1) for r in range(R)], 1)
    return out.numpy(), P.numpy(), ring.numpy()


def band_stats(P, ring, H):
    """per image: mean linear power in rho > 0.5 over mean power in 0 < rho <= 0.25 (DC excluded)."""
    rho = ring / (H / 2)
    hi = P[:, rho > 0.5].mean(1)
    lo = P[:, (rho > 0) & (rho <= 0.25)].mean(1)
    return hi / lo


def slope(ring_logp, H):
    """per image: least-squares slope of ring-mean log10(P/P_DC) against log10(rho), 0 < rho <= 1."""
    R = ring_logp.shape[1]
    rho = np.arange(R) / (H / 2)
    sel = (rho > 0) & (rho <= 1)
    x = np.log10(rho[sel])
    return np.array([np.polyfit(x, y[sel], 1)[0] for y in ring_logp])


def to_bchw(t):
    return t.permute(0, 3, 1, 2) if t.shape[-1] != t.shape[1] and t.dim() == 4 and t.shape[1] == t.shape[2] else t


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--threads', type=int, default=16)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    imgs = np.load(os.path.join(HERE, 'data', 'images_u8.npy'))
    x01 = torch.from_numpy(imgs).permute(0, 3, 1, 2).float() / 255          # (12, 3, 224, 224) in [0,1]
    meta = json.load(open(os.path.join(HERE, 'data', 'images.json')))['images']

    hooks_info, weights_info = {}, {}
    feats = {s: {} for s in STAGES}                                          # stage -> branch -> (12, C, H, W)
    df_routing = {}
    for key in MODELS:
        model = C.Classifier(key)
        wp = C.weight_path(key)
        weights_info[key] = dict(file=os.path.relpath(wp, AN), sha256=sha256(wp), **model.load_info)
        net = model.net
        caps, handles = {}, []
        hooks_info[key] = {}
        for s in STAGES:
            if key == 'caformer_s18' and s != 3:
                continue
            b = block_index(net, s)
            tm = net.stages[s - 1][b].token_mixer
            pre = f'stages.{s - 1}.{b}.token_mixer'
            info = dict(block_index=b, blocks_in_stage=len(net.stages[s - 1]), token_mixer=type(tm).__name__)

            def fwd(name):
                def h(m, i, o):
                    caps[name] = o.detach()
                return h

            def fwd_pre(name):
                def h(m, i):
                    caps[name] = i[0].detach()
                return h
            if key == 'spanetv2_s18_pure':
                handles.append(tm.proj.register_forward_pre_hook(fwd_pre((s, 's_ctx'))))
                info['hooks'] = {f'{pre}.proj (pre-hook, input)': 'SRF outputs s_i of the 4 heads',
                                 f'{pre} (output)': 'mixer output'}
                info['num_heads'] = tm.num_heads
            elif key == 'convformer_s18':
                handles.append(tm.dwconv.register_forward_hook(fwd((s, 'dwconv'))))
                info['hooks'] = {f'{pre}.dwconv (output)': 'depthwise convolution output', f'{pre} (output)': 'mixer output'}
            elif key == 'dfformer_s18':
                handles.append(tm.act2.register_forward_pre_hook(fwd_pre((s, 'filtered'))))
                handles.append(tm.reweight.register_forward_hook(fwd((s, 'routing_logits'))))
                info['hooks'] = {f'{pre}.act2 (pre-hook, input; act2 = {type(tm.act2).__name__})':
                                 'irfft2 output right after the dynamic filter',
                                 f'{pre}.reweight (output)': 'routing logits -> softmax over filters',
                                 f'{pre} (output)': 'mixer output'}
                info['num_filters'] = tm.num_filters
            if key != 'caformer_s18' or s == 3:
                handles.append(tm.register_forward_hook(fwd((s, 'mixer'))))
                if key == 'caformer_s18':
                    info['hooks'] = {f'{pre} (output)': 'attention (token mixer) output'}
            hooks_info[key][s] = info
        with torch.no_grad():
            logits = model(x01)
        weights_info[key]['top1_labels_imagenet'] = [int(i) for i in logits.argmax(1)]
        for hd in handles:
            hd.remove()
        for s in STAGES:
            if (s, 'mixer') not in caps:
                continue
            short = {'spanetv2_s18_pure': 'SPANetV2', 'convformer_s18': 'ConvFormer', 'dfformer_s18': 'DFFormer',
                     'caformer_s18': 'CAFormer'}[key]
            mix = caps[(s, 'mixer')].permute(0, 3, 1, 2).float()
            feats[s][f'{short}|mixer' if key != 'caformer_s18' else 'CAFormer|attention'] = mix
            if key == 'spanetv2_s18_pure':
                sc = caps[(s, 's_ctx')].float()
                B, Cc, H, W = sc.shape
                nh = hooks_info[key][s]['num_heads']
                v = sc.view(B, Cc // nh, nh, H, W)
                for i in range(nh):
                    feats[s][f'SPANetV2|head k={3 + 2 * i}'] = v[:, :, i].contiguous()
            elif key == 'convformer_s18':
                feats[s]['ConvFormer|dwconv'] = caps[(s, 'dwconv')].float()
            elif key == 'dfformer_s18':
                feats[s]['DFFormer|dynamic filter'] = caps[(s, 'filtered')].permute(0, 3, 1, 2).float()
                tm = net.stages[s - 1][hooks_info[key][s]['block_index']].token_mixer
                df_routing[s] = caps[(s, 'routing_logits')].view(x01.shape[0], tm.num_filters, -1).softmax(1).numpy()
        # ---------------- E2 parameters ----------------
        for s in STAGES:
            if key == 'caformer_s18':
                break
            tm = net.stages[s - 1][hooks_info[key][s]['block_index']].token_mixer
            out = {}
            if key == 'convformer_s18':
                out['ConvFormer|dwconv weight'] = tm.dwconv.weight.detach()[:, 0].double().numpy()
            elif key == 'spanetv2_s18_pure':
                for i in range(tm.num_heads):
                    out[f'SPANetV2|k={3 + 2 * i}|kernel'] = getattr(tm, f'local_conv_{i + 1}').weight.detach()[:, 0].double().numpy()
                    out[f'SPANetV2|k={3 + 2 * i}|M'] = getattr(tm, f'filter_{i + 1}').detach().double().numpy()
            elif key == 'dfformer_s18':
                cw = torch.view_as_complex(tm.complex_weights.detach().contiguous()).numpy()    # (H, Wr, F)
                out['DFFormer|basis'] = cw
                out['DFFormer|routing'] = df_routing[s]                                       # (12, F, 2C)
            np.savez_compressed(os.path.join(HERE, 'data', f'e2_params_{key}_stage{s}.npz'), **out)
        del model, net
        print('done', key, flush=True)

    # ---------------- E1 derived data ----------------
    stats = {}
    for s in STAGES:
        maps, spec = {}, {}
        for br, X in feats[s].items():
            maps[br] = X.pow(2).sum(1).sqrt().numpy()
            ring_logp, P, ring = spectrum_rings(X)
            H = X.shape[-1]
            spec[br] = ring_logp
            spec['rho'] = np.arange(ring_logp.shape[1]) / (H / 2)
            hl = band_stats(P, ring, H)
            sl = slope(ring_logp, H)
            stats[f'{s}|{br}'] = dict(stage=s, branch=br, H=H, C=X.shape[1],
                                      hi_lo_power_ratio_mean=float(hl.mean()),
                                      hi_lo_power_ratio_p25=float(np.percentile(hl, 25)),
                                      hi_lo_power_ratio_p75=float(np.percentile(hl, 75)),
                                      slope_mean=float(sl.mean()), slope_p25=float(np.percentile(sl, 25)),
                                      slope_p75=float(np.percentile(sl, 75)))
        np.savez_compressed(os.path.join(HERE, 'data', f'e1_maps_stage{s}.npz'), **maps)
        np.savez_compressed(os.path.join(HERE, 'data', f'e1_spectra_stage{s}.npz'), **spec)
    json.dump(stats, open(os.path.join(HERE, 'data', 'e1_stats.json'), 'w'), indent=1)
    json.dump(dict(hooks=hooks_info, weights=weights_info, torch=torch.__version__,
                   images=[m['file'] for m in meta]), open(os.path.join(HERE, 'data', 'hooks.json'), 'w'), indent=1,
              default=str)
    print('E1 / E2 data written')


if __name__ == '__main__':
    main()
