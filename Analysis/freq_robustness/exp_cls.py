#!/usr/bin/env python
"""Experiments 0, A, B, C for one classification model (inference only; GPU chosen with CUDA_VISIBLE_DEVICES).

    python exp_cls.py clean   --model spanetv2_s18_pure [--weight PATH]   # 0: clean top-1, full val (fp32)
    python exp_cls.py band    --model ...                                 # A: band-filtered val, full 50k (fp32)
    python exp_cls.py fourier --model ... [--amp]                         # B: Fourier heat map, 5k subset
    python exp_cls.py fourier_sparse --model ...                          # B': stride-8 grid outside the 63x63 window
    python exp_cls.py pgd     --model ...                                 # C: band-limited L2 PGD, 5k subset (fp32)

Weights: --weight overrides the path recorded in data/weights_manifest.json. Raw results go to data/raw/.
All perturbations act on the [0,1] pixel tensor of the model's own eval pipeline (see common.py); only C clamps.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import common as C

RAW = os.path.join(C.DATA, 'raw')

# ---- A: Bai et al. (ECCV 2022) eq. (1)-(2), Fig. 2 -------------------------------------------------------------
BAND_LOW_S = list(range(20, 221, 20))          # 20, 40, ..., 220
BAND_HIGH_S = list(range(170, 221, 10))        # 170, 180, ..., 220
MASK_METRIC = 'chebyshev'                      # square masks: HAT released no filtering code (report.md A)

# ---- B: Yin et al. (NeurIPS 2019) Fourier basis noise -------------------------------------------------------
FOURIER_HALF = 31                              # 63 x 63 centre window: offsets -31..31
FOURIER_NORM = 15.7                            # l2 norm v per channel, [0,1] pixels (Yin et al.; Bai et al. Fig. 7)
FOURIER_PER_CHANNEL_SIGN = True                # random sign r drawn per channel (Yin et al.: channels perturbed independently)

# ---- C: band-limited L2 PGD ----------------------------------------------------------------------------------
PGD_EPS = [0.25, 0.5, 1.0]
PGD_STEPS = 10
PGD_LOW_S = 56


def save(name, obj):
    os.makedirs(RAW, exist_ok=True)
    with open(os.path.join(RAW, name), 'w') as f:
        json.dump(obj, f, indent=1)


@torch.no_grad()
def run_clean(model, cache, dev, bs, amp=False):
    idx = np.arange(len(cache.y))
    pred = np.empty(len(idx), np.int64)
    for j, xb, yb in cache.batches(idx, bs):
        x = C.to01(xb, dev)
        with torch.autocast('cuda', torch.float16, enabled=amp):
            pred[j] = model(x).argmax(1).cpu().numpy()
    return pred


def exp_clean(args, model, cache, dev):
    t0 = time.time()
    pred = run_clean(model, cache, dev, args.bs)
    acc = float((pred == cache.y).mean() * 100)
    sub = C.subset5k()
    pred16 = run_clean(model, cache, dev, args.bs, amp=True)
    out = dict(model=args.model, top1=acc, n=len(pred), subset5k_top1=float((pred[sub] == cache.y[sub]).mean() * 100),
               fp16_top1=float((pred16 == cache.y).mean() * 100), fp16_fp32_pred_agreement=float((pred16 == pred).mean()),
               load_info=model.load_info, preprocessing=C.preprocessing(args.model), seconds=time.time() - t0,
               n_params=sum(p.numel() for p in model.net.parameters()))
    np.save(os.path.join(RAW, f'clean_pred_{args.model}.npy'), pred)
    save(f'clean_{args.model}.json', out)
    print(json.dumps({k: v for k, v in out.items() if k != 'load_info'}), flush=True)


@torch.no_grad()
def exp_band(args, model, cache, dev):
    masks = [('low', S, C.lowpass_mask(S, metric=MASK_METRIC).to(dev)) for S in BAND_LOW_S] + \
            [('high', S, C.highpass_mask(S, metric=MASK_METRIC).to(dev)) for S in BAND_HIGH_S]
    correct = {f'{k}{S}': 0 for k, S, _ in masks}
    idx = np.arange(len(cache.y))
    t0 = time.time()
    for j, xb, yb in cache.batches(idx, args.bs):
        x = C.to01(xb, dev)
        y = torch.from_numpy(yb).to(dev)
        Fx = C.shifted_fft(x)
        for k, S, m in masks:
            correct[f'{k}{S}'] += int((model(C.apply_mask(x, m, Fx)).argmax(1) == y).sum())
    out = dict(model=args.model, n=len(idx), metric=MASK_METRIC, seconds=time.time() - t0,
               top1={k: v / len(idx) * 100 for k, v in correct.items()})
    save(f'band_{args.model}.json', out)
    print(json.dumps(out['top1']), flush=True)


def fourier_basis(u, v, n=224):
    """Yin et al. (NeurIPS 2019) Fourier basis U_{u,v}, following google-research frequency_analysis/freq_heatmap.py:
    fftshifted spectrum 0.5+0.5j at (c+u, c+v) and 0.5-0.5j at the symmetric coordinate (c-u, c-v), inverse FFT,
    real part, then scaled to unit l2 norm (the official code scales by sqrt(H*W), which gives norm 1 except at
    self-symmetric frequencies such as DC; here the norm is set to 1 explicitly for every frequency)."""
    c = n // 2
    spec = torch.zeros(n, n, dtype=torch.complex128)
    spec[(c + u) % n, (c + v) % n] = 0.5 + 0.5j
    spec[(c - u) % n, (c - v) % n] = 0.5 - 0.5j
    b = torch.fft.ifft2(torch.fft.ifftshift(spec)).real
    return (b / b.norm()).float()


FOURIER_SPARSE_STRIDE = 8


def sparse_coords(half=FOURIER_HALF, stride=FOURIER_SPARSE_STRIDE, n=224):
    """Frequencies on a stride-8 grid over the whole spectrum, outside the 63x63 window, one per conjugate pair
    (u in 0..n/2, v in -n/2+stride..n/2; u == 0 or u == n/2 rows keep v >= 0)."""
    c = n // 2
    out = []
    for u in range(0, c + 1, stride):
        for v in range(-c + stride, c + 1, stride):
            if (u == 0 or u == c) and v < 0:
                continue
            if abs(u) <= half and abs(v) <= half:
                continue
            out.append((u, v))
    return out


@torch.no_grad()
def exp_fourier(args, model, cache, dev, sparse=False):
    sub = C.subset5k()
    X = cache.x[sub]
    Y = torch.from_numpy(cache.y[sub]).to(dev)
    H = FOURIER_HALF
    # half plane: (u, v) with u > 0, or u == 0 and v >= 0 ; the other half is filled by symmetry
    coords = [(u, v) for u in range(0, H + 1) for v in range(-H, H + 1) if u > 0 or v >= 0]
    if sparse:
        coords = sparse_coords()
    gen = torch.Generator().manual_seed(0)
    err = {}
    t0 = time.time()
    xs = [C.to01(X[i:i + args.bs], dev) for i in range(0, len(sub), args.bs)]   # 5k x 3 x 224 x 224 fp32 = 3 GB
    for ci, (u, v) in enumerate(coords):
        b = fourier_basis(u, v).to(dev)
        wrong = 0
        for i, x in enumerate(xs):
            shape = (x.shape[0], 3, 1, 1) if FOURIER_PER_CHANNEL_SIGN else (x.shape[0], 1, 1, 1)
            r = (torch.randint(0, 2, shape, generator=gen) * 2 - 1).float().to(dev)
            xp = x + r * FOURIER_NORM * b                # no clamp
            with torch.autocast('cuda', torch.float16, enabled=args.amp):
                wrong += int((model(xp).argmax(1) != Y[i * args.bs:i * args.bs + x.shape[0]]).sum())
        err[f'{u},{v}'] = wrong / len(sub)
        if ci % 100 == 0:
            print(f'{ci}/{len(coords)} ({u},{v}) err {err[f"{u},{v}"]:.3f} {time.time() - t0:.0f}s', flush=True)
    if sparse:
        save(f'fourier_sparse_{args.model}.json', dict(model=args.model, n=len(sub), norm=FOURIER_NORM,
                                                       stride=FOURIER_SPARSE_STRIDE, amp=args.amp, error=err,
                                                       per_channel_sign=FOURIER_PER_CHANNEL_SIGN,
                                                       seconds=time.time() - t0))
        return
    grid = np.full((2 * H + 1, 2 * H + 1), np.nan)
    for k, e in err.items():
        u, v = map(int, k.split(','))
        grid[H + u, H + v] = e
        grid[H - u, H - v] = e
    np.save(os.path.join(RAW, f'fourier_{args.model}.npy'), grid)
    save(f'fourier_{args.model}.json', dict(model=args.model, n=len(sub), norm=FOURIER_NORM, half=H, amp=args.amp,
                                            per_channel_sign=FOURIER_PER_CHANNEL_SIGN, seconds=time.time() - t0))


def band_project(d, mask_half):
    """d (B,3,H,W) -> irfft2(rfft2(d, ortho) * M, ortho): M is a real, conjugate-symmetric mask (rfft half grid)."""
    return torch.fft.irfft2(torch.fft.rfft2(d, norm='ortho') * mask_half, s=d.shape[-2:], norm='ortho')


def half_mask(full_shifted):
    """fftshifted (H,W) mask -> unshifted rfft half grid (H, W//2+1)."""
    m = torch.fft.ifftshift(full_shifted, dim=(-2, -1))
    return m[:, : m.shape[1] // 2 + 1]


def l2n(t):
    return t.flatten(1).norm(dim=1).view(-1, 1, 1, 1)


def pgd(model, x, y, eps, mask_half, steps=PGD_STEPS):
    """Untargeted L2 PGD, no random start, step eps/4. Each step: gradient step -> band projection -> L2 ball ->
    [0,1] clamp of x + delta. Returns x_adv and the fraction of delta's energy outside the band after the clamp."""
    alpha = eps / 4
    d = torch.zeros_like(x)
    for _ in range(steps):
        d.requires_grad_(True)
        loss = F.cross_entropy(model(x + d), y)
        g, = torch.autograd.grad(loss, d)
        with torch.no_grad():
            d = d + alpha * g / l2n(g).clamp_min(1e-12)
            if mask_half is not None:
                d = band_project(d, mask_half)
            d = d * torch.clamp(eps / l2n(d).clamp_min(1e-12), max=1.0)
            d = (x + d).clamp(0, 1) - x
    with torch.no_grad():
        leak = None
        if mask_half is not None:
            inb = band_project(d, mask_half)
            leak = ((d - inb).flatten(1).norm(dim=1) ** 2 / (d.flatten(1).norm(dim=1) ** 2).clamp_min(1e-20))
    return (x + d).detach(), d.detach(), leak


def exp_pgd(args, model, cache, dev):
    C.set_fp32()
    sub = C.subset5k()
    low = C.lowpass_mask(PGD_LOW_S, metric=MASK_METRIC)
    variants = {'full': None, 'low': half_mask(low).to(dev), 'high': half_mask(1 - low).to(dev)}
    res = {}
    t0 = time.time()
    for name, mh in variants.items():
        for eps in PGD_EPS:
            correct, leaks, norms = 0, [], []
            for j, xb, yb in cache.batches(sub, args.bs):
                x = C.to01(xb, dev)
                y = torch.from_numpy(yb).to(dev)
                xa, d, leak = pgd(model, x, y, eps, mh)
                with torch.no_grad():
                    correct += int((model(xa).argmax(1) == y).sum())
                norms.append(l2n(d).flatten().cpu())
                if leak is not None:
                    leaks.append(leak.cpu())
            nr = torch.cat(norms)
            res[f'{name}|{eps}'] = dict(top1=correct / len(sub) * 100, delta_l2_mean=float(nr.mean()),
                                        delta_l2_max=float(nr.max()),
                                        out_of_band_energy_mean=float(torch.cat(leaks).mean()) if leaks else None,
                                        out_of_band_energy_max=float(torch.cat(leaks).max()) if leaks else None)
            print(name, eps, json.dumps(res[f'{name}|{eps}']), f'{time.time() - t0:.0f}s', flush=True)
    with torch.no_grad():
        correct = sum(int((model(C.to01(xb, dev)).argmax(1) == torch.from_numpy(yb).to(dev)).sum())
                      for _, xb, yb in cache.batches(sub, args.bs))
    save(f'pgd_{args.model}.json', dict(model=args.model, n=len(sub), clean_top1=correct / len(sub) * 100,
                                        steps=PGD_STEPS, low_S=PGD_LOW_S, metric=MASK_METRIC, results=res,
                                        seconds=time.time() - t0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('exp', choices=['clean', 'band', 'fourier', 'fourier_sparse', 'pgd'])
    ap.add_argument('--model', required=True, choices=list(C.MODELS))
    ap.add_argument('--weight', default=None)
    ap.add_argument('--bs', type=int, default=None, help='default 250; 50 for pgd (fp32 backward through the model)')
    ap.add_argument('--amp', action='store_true', help='fp16 autocast (fourier only)')
    args = ap.parse_args()
    if args.bs is None:
        args.bs = 50 if args.exp == 'pgd' else 250       # batch 250 with a backward pass exceeds 24 GB
    os.makedirs(RAW, exist_ok=True)
    dev = torch.device('cuda')
    C.set_fp32()
    model = C.Classifier(args.model, args.weight).to(dev)
    cache = C.ValCache(C.MODELS[args.model]['crop'])
    {'clean': exp_clean, 'band': exp_band, 'fourier': exp_fourier, 'pgd': exp_pgd,
     'fourier_sparse': lambda *a: exp_fourier(*a, sparse=True)}[args.exp](args, model, cache, dev)


if __name__ == '__main__':
    main()
