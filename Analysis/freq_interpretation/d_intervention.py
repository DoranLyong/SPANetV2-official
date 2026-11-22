#!/usr/bin/env python
"""D (optional). Inference-time intervention on the SRF of SPANetV2-S18-pure / -hybrid (fp32, GPU).

    python d_intervention.py --model spanetv2_s18_pure --intervene {none,flat}

'flat': in the loaded model (in memory only; no code or file is changed), every SRF parameter M of every ExSPAM block is
replaced by the constant logit(mean_f psi_c(f)) per channel, with the conjugate-symmetric full-grid mean of
psi = sigmoid(M) (srf_task_analysis Grid). The overall gain of each channel stays, its frequency shape is removed.
The modified model was never trained.
Measures: clean top-1 on the full ImageNet val (50,000); on the 1,000 images: Fourier heat map on the 63x63 window with
stride 2 (offsets -30..30 step 2, one per conjugate pair; freq_robustness noise: l2 15.7 per channel, random sign per
image and channel), and high-band L2 PGD eps 1.0 with 10 steps (step eps/4, as in freq_robustness C).
"""
import argparse
import time

import numpy as np
import torch

import fi_common as FC
from fi_common import C, E, SRF
import a_pgd as A

BS = 250


def flatten_srf(net):
    n = 0
    for stage in net.stages:
        for blk in stage:
            tm = blk.token_mixer
            if type(tm).__name__ != 'ExSPAM':
                continue
            for h in range(tm.num_heads):
                P = getattr(tm, f'filter_{h + 1}')
                psi = torch.sigmoid(P.detach()).double().cpu().numpy()
                g = SRF.Grid(*psi.shape[1:])
                m = g.wmean(psi)                                              # (C4,)
                with torch.no_grad():
                    P.copy_(torch.from_numpy(np.log(m / (1 - m))).float().view(-1, 1, 1).expand_as(P))
                n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=['spanetv2_s18_pure', 'spanetv2_s18_hybrid'])
    ap.add_argument('--intervene', required=True, choices=['none', 'flat'])
    args = ap.parse_args()
    C.set_fp32()
    torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = False, True
    dev = torch.device('cuda')
    model = C.Classifier(args.model)
    n_srf = flatten_srf(model.net) if args.intervene == 'flat' else 0
    model = model.to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    cache = C.ValCache(1.0)
    t0 = time.time()
    correct = 0
    with torch.no_grad():
        for j, xb, yb in cache.batches(np.arange(len(cache.y)), BS):
            correct += int((model(C.to01(xb, dev)).argmax(1).cpu().numpy() == yb).sum())
    clean_full = correct / len(cache.y) * 100
    idx = FC.subset1k()
    X, Y = np.array(cache.x[idx]), cache.y[idx]
    xs = [C.to01(X[i:i + BS], dev) for i in range(0, len(X), BS)]
    ys = [torch.from_numpy(Y[i:i + BS]).to(dev) for i in range(0, len(X), BS)]
    with torch.no_grad():
        clean_1k = float(np.mean(np.concatenate([(model(x).argmax(1) == y).cpu().numpy() for x, y in zip(xs, ys)])) * 100)
    offs = list(range(-30, 31, 2))
    coords = [(u, v) for u in offs if u >= 0 for v in offs if u > 0 or v >= 0]
    gen = torch.Generator().manual_seed(0)
    err = {}
    with torch.no_grad():
        for (u, v) in coords:
            b = E.fourier_basis(u, v).to(dev)
            wrong = 0
            for x, y in zip(xs, ys):
                r = (torch.randint(0, 2, (x.shape[0], 3, 1, 1), generator=gen) * 2 - 1).float().to(dev)
                wrong += int((model(x + r * E.FOURIER_NORM * b).argmax(1) != y).sum())
            err[f'{u},{v}'] = wrong / len(X) * 100
    rho = {k: np.hypot(*map(int, k.split(','))) / 112 for k in err}
    mh = FC.masks(dev)['high']
    corr = []
    for i in range(0, len(X), 50):
        x = C.to01(X[i:i + 50], dev)
        y = torch.from_numpy(Y[i:i + 50]).to(dev)
        c, _, _ = A.attack(model, x, y, 1.0, mh, 10, 0.25)
        corr.append(c)
    pgd = float(np.concatenate(corr).mean() * 100)
    e = np.array(list(err.values()))
    lo = np.array([err[k] for k in err if rho[k] <= 0.25 + 1e-9])
    FC.save_json(f'd_{args.model}_{args.intervene}.json', dict(
        model=args.model, intervene=args.intervene, n_srf_tensors_changed=n_srf, clean_top1_val50k=clean_full,
        clean_top1_1k=clean_1k, fourier_mean_error=float(e.mean()), fourier_low_error=float(lo.mean()),
        fourier_rest_error=float(np.mean([err[k] for k in err if rho[k] > 0.25 + 1e-9])),
        fourier_mean_error_minus_clean=float(e.mean() - (100 - clean_1k)), fourier_errors=err,
        pgd_high_eps1_10step_top1=pgd, seconds=time.time() - t0))
    print(args.model, args.intervene, clean_full, clean_1k, e.mean(), pgd, flush=True)


if __name__ == '__main__':
    main()
