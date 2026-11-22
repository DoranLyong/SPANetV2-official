#!/usr/bin/env python
"""B. Where does a perturbation grow inside the network? (1,000 images, fp32, GPU)

    python b_propagation.py --model M

Perturbations (input-space, on the [0,1] pixels of the model's pipeline):
  fourier_top_i / fourier_bottom_i : the 5 frequencies with the highest / lowest error of ConvFormer-S18's Fourier heat
      map (freq_robustness data/raw/fourier_convformer_s18.npy, one per conjugate pair), noise x + r * 15.7 * U with
      the basis and random per-image/per-channel sign of freq_robustness exp_cls (no clamp; seed = frequency index)
  pgd_own     : high-band L2 PGD, eps 1.0, 100 steps (a_pgd.attack, step 2.5 eps / 100, no random start) on the model
                itself (the A deltas in cache/deltas, computed here for DFFormer / CDFFormer)
  pgd_caformer: the CAFormer-S18 delta of the same setting, added to every model (same input change for all)
Measurement points (forward hooks): stem (downsample_layers.0 output), every block's token-mixer branch output
(token_mixer output, before the residual addition) and MLP branch output (mlp output), every stage output.
Quantities per point and image: r = ||h(x + d) - h(x)|| / ||h(x)||, and the share of ||dh||^2 at high frequencies:
dh (B, H, W, C) -> 2D DFT over (H, W) -> energy outside the centred square |ky|, |kx| <= H/8 (the same band split
as M_l^56 on the 224 grid, scaled to the resolution of the point) over total energy.
"""
import argparse
import os

import numpy as np
import torch

import fi_common as FC
from fi_common import C, E
import a_pgd as A

BS = 50


def fourier_freqs(k=5):
    g = np.load(os.path.join(FC.FR, 'data', 'raw', 'fourier_convformer_s18.npy'))
    H = g.shape[0] // 2
    cells = [(u, v) for u in range(0, H + 1) for v in range(-H, H + 1) if u > 0 or v >= 0]
    err = np.array([g[H + u, H + v] for u, v in cells])
    o = np.argsort(err)
    return [cells[i] for i in o[::-1][:k]], [cells[i] for i in o[:k]], {f'{u},{v}': float(e) for (u, v), e in zip(cells, err)}


def points(net):
    out = [('stem', 0, -1, net.downsample_layers[0])]
    for s, stage in enumerate(net.stages):
        for b, blk in enumerate(stage):
            out.append(('mixer', s + 1, b, blk.token_mixer))
            out.append(('mlp', s + 1, b, blk.mlp))
        out.append(('stage', s + 1, -1, stage))
    return out


def high_share(dh):
    """dh (B, H, W, C) -> (B,) energy share outside |ky|, |kx| <= H/8."""
    B, H, W, Cc = dh.shape
    P = (torch.fft.fft2(dh.permute(0, 3, 1, 2).double(), norm='ortho').abs() ** 2).sum(1)      # (B, H, W)
    iy = torch.fft.fftfreq(H, 1 / H, device=dh.device).abs().view(-1, 1)
    ix = torch.fft.fftfreq(W, 1 / W, device=dh.device).abs().view(1, -1)
    low = (iy <= H / 8) & (ix <= W / 8)
    return (P[:, ~low].sum(1) / P.sum((1, 2))).float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    args = ap.parse_args()
    C.set_fp32()
    torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = False, True
    dev = torch.device('cuda')
    model = C.Classifier(args.model).to(dev)
    for p in model.parameters():
        p.requires_grad_(False)
    idx = FC.subset1k()
    cache = C.ValCache(1.0)
    X, Y = np.array(cache.x[idx]), cache.y[idx]
    top, bottom, err = fourier_freqs()
    pert_names = [f'fourier_top_{i}' for i in range(5)] + [f'fourier_bottom_{i}' for i in range(5)] + ['pgd_own', 'pgd_caformer']
    basis = {f'fourier_top_{i}': E.fourier_basis(*top[i]).to(dev) for i in range(5)}
    basis.update({f'fourier_bottom_{i}': E.fourier_basis(*bottom[i]).to(dev) for i in range(5)})
    dpath = lambda m: os.path.join(FC.CACHE, 'deltas', f'{m}_high_eps1.0_s100.npy')
    if not os.path.exists(dpath(args.model)):          # DFFormer / CDFFormer: same attack as A
        mh = FC.masks(dev)['high']
        buf = np.empty((len(X), 3, 224, 224), np.float16)
        for i in range(0, len(X), BS):
            x = C.to01(X[i:i + BS], dev)
            y = torch.from_numpy(Y[i:i + BS]).to(dev)
            _, _, d = A.attack(model, x, y, 1.0, mh, 100, 2.5 / 100)
            buf[i:i + BS] = d.cpu().numpy().astype(np.float16)
        os.makedirs(os.path.dirname(dpath(args.model)), exist_ok=True)
        np.save(dpath(args.model), buf)
    D_own = np.load(dpath(args.model), mmap_mode='r')
    D_caf = np.load(dpath('caformer_s18'), mmap_mode='r')
    pts = points(model.net)
    caps = {}
    hooks = [m.register_forward_hook(lambda mod, i, o, k=k: caps.__setitem__(k, o.detach()))
             for k, (_, _, _, m) in enumerate(pts)]
    R = {n: [[] for _ in pts] for n in pert_names}
    HS = {n: [[] for _ in pts] for n in pert_names}
    correct = {n: [] for n in pert_names + ['clean']}
    for i in range(0, len(X), BS):
        x = C.to01(X[i:i + BS], dev)
        y = torch.from_numpy(Y[i:i + BS]).to(dev)
        with torch.no_grad():
            caps.clear()
            correct['clean'].append((model(x).argmax(1) == y).cpu().numpy())
            clean = dict(caps)
            for n in pert_names:
                if n.startswith('fourier'):
                    g = torch.Generator().manual_seed(pert_names.index(n) * 100000 + i)
                    r = (torch.randint(0, 2, (x.shape[0], 3, 1, 1), generator=g) * 2 - 1).float().to(dev)
                    xp = x + r * E.FOURIER_NORM * basis[n]
                else:
                    D = D_own if n == 'pgd_own' else D_caf
                    xp = (x + torch.from_numpy(np.array(D[i:i + BS])).float().to(dev)).clamp(0, 1)
                caps.clear()
                correct[n].append((model(xp).argmax(1) == y).cpu().numpy())
                for k in range(len(pts)):
                    h0, h1 = clean[k], caps[k]
                    dh = h1 - h0
                    R[n][k].append((dh.flatten(1).norm(dim=1) / h0.flatten(1).norm(dim=1)).cpu().numpy())
                    HS[n][k].append(high_share(dh).cpu().numpy())
        print(args.model, i + BS, flush=True)
    for h in hooks:
        h.remove()
    out = dict(model=args.model, n=len(X), fourier_top=[f'{u},{v}' for u, v in top], fourier_bottom=[f'{u},{v}' for u, v in bottom],
               fourier_err_convformer={f'{u},{v}': err[f'{u},{v}'] for u, v in top + bottom},
               top1={n: float(np.concatenate(v).mean() * 100) for n, v in correct.items()}, points=[])
    for k, (kind, s, b, _) in enumerate(pts):
        rec = dict(kind=kind, stage=s, block=b)
        for n in pert_names:
            r = np.concatenate(R[n][k])
            hs = np.concatenate(HS[n][k])
            rec[n] = dict(r_mean=float(r.mean()), r_p25=float(np.percentile(r, 25)), r_p75=float(np.percentile(r, 75)),
                          high_share_mean=float(hs.mean()))
        out['points'].append(rec)
    FC.save_json(f'b_{args.model}.json', out)


if __name__ == '__main__':
    main()
