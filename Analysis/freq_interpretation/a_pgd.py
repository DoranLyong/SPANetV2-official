#!/usr/bin/env python
"""A. Reliability of the band-limited PGD results (1,000 images, fp32, GPU via CUDA_VISIBLE_DEVICES).

    python a_pgd.py main --model M --variant {high,full}   # eps 0.5/1.0; 10/50/100 steps; 5 random restarts @100
    python a_pgd.py large --model M                       # full, eps 2/4/8, 100 steps
    python a_pgd.py apgd --model M                        # autoattack APGD-CE (L2), full, eps 0.5/1.0, 100 iter
    python a_pgd.py transfer                              # CAFormer<->hybrid, ConvFormer<->pure deltas, eps 1.0
    python a_pgd.py apgd2 --model M --variant {high,full} # APGD-CE (L2) with band projection: high eps 0.25/0.5/1.0,
                                                          # full eps 0.25 (per-image results stored)

Band-limited APGD (apgd2): autoattack's APGDAttack, unchanged, with one method overridden in a subclass:
normalize(t) = band_project(t) / ||band_project(t)||_2. In the L2 branch of APGD this function produces the random start
direction, the step direction (projected gradient) and the direction of x_adv - x before the eps-ball scaling and the
[0,1] clamp, so every iterate is (step -> band projection -> L2 ball -> clamp), as in the PGD here. 'full' uses no mask,
i.e. the original APGDAttack. The share of the final delta's energy outside the band is recorded.

PGD = freq_robustness exp_cls.pgd with a configurable step count / step size / random start: each step
grad of CE -> d += alpha * g / ||g||_2 -> band projection (irfft2(rfft2(d, ortho) * M, ortho), M = M_l^56 complement
for 'high', none for 'full') -> L2 ball -> d = clamp(x + d, 0, 1) - x. Steps 10 use alpha = eps/4 (= the original
setting), steps 50 / 100 use alpha = 2.5 eps / steps. Random start: Gaussian direction, band-projected, radius
eps * u^(1/n) (u ~ U(0,1), n = 3*224*224), then clamped; 5 restarts with seeds 0-4, an image counts as robust only if it
is classified correctly after every restart. Accuracy is that of the final iterate. Mean CE over the images is
recorded at every step. cuDNN runs deterministically (benchmark off); with the default non-deterministic kernels the
same attack gives slightly different results on every run, which is measured separately (two extra 10-step runs at
eps 1.0). The 100-step, eps 1.0, no-random-start deltas are stored in cache/deltas (float16) for the
transfer attacks and for B.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

import fi_common as FC
from fi_common import C, E

AUTOATTACK = os.path.join(FC.HERE, 'third_party', 'auto-attack')
MODELS = ['spanetv2_s18_hybrid', 'caformer_s18', 'spanetv2_s18_pure', 'convformer_s18', 'deit_small']
BS = 50
DET = [True]


def attack(model, x, y, eps, mh, steps, alpha, rand_seed=None):
    n = x[0].numel()
    if rand_seed is None:
        d = torch.zeros_like(x)
    else:
        g = torch.Generator(device='cpu').manual_seed(int(rand_seed))
        d = torch.randn(x.shape, generator=g).to(x.device)
        if mh is not None:
            d = E.band_project(d, mh)
        u = torch.rand(x.shape[0], 1, 1, 1, generator=g).to(x.device)
        d = d / E.l2n(d).clamp_min(1e-12) * eps * u ** (1.0 / n)
        d = (x + d).clamp(0, 1) - x
    losses = []
    for _ in range(steps):
        d.requires_grad_(True)
        lv = F.cross_entropy(model(x + d), y, reduction='none')
        g, = torch.autograd.grad(lv.mean(), d)          # mean, as in freq_robustness exp_cls.pgd
        losses.append(float(lv.detach().sum()))
        with torch.no_grad():
            d = d + alpha * g / E.l2n(g).clamp_min(1e-12)
            if mh is not None:
                d = E.band_project(d, mh)
            d = d * torch.clamp(eps / E.l2n(d).clamp_min(1e-12), max=1.0)
            d = (x + d).clamp(0, 1) - x
    with torch.no_grad():
        logits = model(x + d)
        losses.append(float(F.cross_entropy(logits, y, reduction='sum')))
        correct = (logits.argmax(1) == y)
    return correct.cpu().numpy(), np.array(losses), d.detach()


def run_batches(model, X, Y, dev, fn):
    outs = []
    for i in range(0, len(X), BS):
        x = C.to01(X[i:i + BS], dev)
        y = torch.from_numpy(Y[i:i + BS]).to(dev)
        outs.append(fn(x, y, i))
    return outs


def part_main(args, model, X, Y, dev):
    mh = FC.masks(dev)[args.variant]
    res = {}
    for eps in (0.5, 1.0):
        for steps in (10, 50, 100):
            alpha = eps / 4 if steps == 10 else 2.5 * eps / steps
            keep = steps == 100 and eps == 1.0
            dbuf = np.empty((len(X), 3, 224, 224), np.float16) if keep else None

            def fn(x, y, i):
                c, l, d = attack(model, x, y, eps, mh, steps, alpha)
                if keep:
                    dbuf[i:i + len(x)] = d.cpu().numpy().astype(np.float16)
                return c, l
            t0 = time.time()
            o = run_batches(model, X, Y, dev, fn)
            corr = np.concatenate([c for c, _ in o])
            loss = np.sum([l for _, l in o], 0) / len(X)
            res[f'{eps}|{steps}|norand'] = dict(top1=float(corr.mean() * 100), alpha=alpha, loss_curve=loss.tolist(),
                                                correct=corr.astype(int).tolist(), seconds=time.time() - t0)
            print(args.model, args.variant, eps, steps, 'norand', f'{corr.mean() * 100:.2f}', f'{time.time() - t0:.0f}s',
                  flush=True)
            if keep:
                os.makedirs(os.path.join(FC.CACHE, 'deltas'), exist_ok=True)
                np.save(os.path.join(FC.CACHE, 'deltas', f'{args.model}_{args.variant}_eps1.0_s100.npy'), dbuf)
        if eps == 1.0:      # run-to-run variation of the original 10-step attack with non-deterministic GPU kernels
            torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = False, True
            reps = []
            for _ in range(2):
                o = run_batches(model, X, Y, dev, lambda x, y, i: attack(model, x, y, eps, mh, 10, eps / 4)[:2])
                reps.append(float(np.concatenate([c for c, _ in o]).mean() * 100))
            torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = DET[0], not DET[0]
            res['1.0|10|norand|nondeterministic_repeats'] = dict(top1=reps)
            print(args.model, args.variant, 'nondeterministic repeats', reps, flush=True)
        steps, alpha = 100, 2.5 * eps / 100
        allc, curves = [], []
        t0 = time.time()
        for r in range(5):
            o = run_batches(model, X, Y, dev, lambda x, y, i, r=r: attack(model, x, y, eps, mh, steps, alpha,
                                                                         rand_seed=1000 * r + i)[:2])
            allc.append(np.concatenate([c for c, _ in o]))
            curves.append((np.sum([l for _, l in o], 0) / len(X)).tolist())
        worst = np.all(np.stack(allc), 0)
        res[f'{eps}|100|rand5'] = dict(top1=float(worst.mean() * 100), per_restart_top1=[float(c.mean() * 100) for c in allc],
                                       alpha=alpha, loss_curves=curves, correct=worst.astype(int).tolist(),
                                       seconds=time.time() - t0)
        print(args.model, args.variant, eps, 'rand5 worst', f'{worst.mean() * 100:.2f}', flush=True)
    FC.save_json(f'a_main_{args.model}_{args.variant}.json', dict(model=args.model, variant=args.variant, n=len(X),
                                                                   deterministic_cudnn=DET[0], results=res))


def part_large(args, model, X, Y, dev):
    res = {}
    for eps in (2.0, 4.0, 8.0):
        o = run_batches(model, X, Y, dev, lambda x, y, i, eps=eps: attack(model, x, y, eps, None, 100, 2.5 * eps / 100)[:2])
        corr = np.concatenate([c for c, _ in o])
        res[str(eps)] = dict(top1=float(corr.mean() * 100), loss_curve=(np.sum([l for _, l in o], 0) / len(X)).tolist())
        print(args.model, 'large', eps, f'{corr.mean() * 100:.2f}', flush=True)
    FC.save_json(f'a_large_{args.model}.json', dict(model=args.model, variant='full', steps=100, n=len(X), results=res))


def part_apgd(args, model, X, Y, dev):
    sys.path.insert(0, AUTOATTACK)
    from autoattack.autopgd_base import APGDAttack
    import subprocess
    commit = subprocess.run(['git', '-C', AUTOATTACK, 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    res = {}
    for eps in (0.5, 1.0):
        atk = APGDAttack(model, n_iter=100, norm='L2', n_restarts=1, eps=eps, seed=0, loss='ce', eot_iter=1, rho=.75,
                         device=dev)

        def fn(x, y, i):
            xa = atk.perturb(x.contiguous(), y)          # autoattack uses .view on the gradient
            with torch.no_grad():
                return (model(xa).argmax(1) == y).cpu().numpy(), float((xa - x).flatten(1).norm(dim=1).max())
        o = run_batches(model, X, Y, dev, fn)
        corr = np.concatenate([c for c, _ in o])
        res[str(eps)] = dict(top1=float(corr.mean() * 100), max_l2=max(m for _, m in o))
        print(args.model, 'apgd', eps, f'{corr.mean() * 100:.2f}', flush=True)
    FC.save_json(f'a_apgd_{args.model}.json', dict(model=args.model, variant='full', attack='APGD-CE L2, 100 iter, 1 restart',
                                                   autoattack_commit=commit, n=len(X), results=res))


def part_apgd2(args, model, X, Y, dev):
    sys.path.insert(0, AUTOATTACK)
    from autoattack.autopgd_base import APGDAttack
    import subprocess
    commit = subprocess.run(['git', '-C', AUTOATTACK, 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    mh = FC.masks(dev)[args.variant]

    class BandAPGD(APGDAttack):
        def normalize(self, t):
            if mh is not None:
                t = E.band_project(t, mh)
            return super().normalize(t.contiguous())

    res = {}
    for eps in ((0.25, 0.5, 1.0) if args.variant == 'high' else (0.25,)):
        atk = BandAPGD(model, n_iter=100, norm='L2', n_restarts=1, eps=eps, seed=0, loss='ce', eot_iter=1, rho=.75,
                       device=dev)

        def fn(x, y, i):
            xa = atk.perturb(x.contiguous(), y)
            d = xa - x
            with torch.no_grad():
                c = (model(xa).argmax(1) == y).cpu().numpy()
            out = 0.0
            if mh is not None:
                out = float(((d - E.band_project(d, mh)).flatten(1).norm(dim=1) ** 2 /
                             d.flatten(1).norm(dim=1).clamp_min(1e-12) ** 2).mean())
            return c, float(d.flatten(1).norm(dim=1).max()), out
        t0 = time.time()
        o = run_batches(model, X, Y, dev, fn)
        corr = np.concatenate([c for c, _, _ in o])
        res[str(eps)] = dict(top1=float(corr.mean() * 100), max_l2=max(m for _, m, _ in o),
                             out_of_band_energy_share=float(np.mean([q for _, _, q in o])),
                             correct=corr.astype(int).tolist(), seconds=time.time() - t0)
        print(args.model, 'apgd2', args.variant, eps, f'{corr.mean() * 100:.2f}', f'{time.time() - t0:.0f}s', flush=True)
    FC.save_json(f'a_apgd2_{args.model}_{args.variant}.json', dict(
        model=args.model, variant=args.variant, attack='APGD-CE L2, 100 iter, 1 restart, band-projected normalize',
        autoattack_commit=commit, deterministic_cudnn=DET[0], n=len(X), results=res))


def part_transfer(args, dev):
    X, Y = load_images()
    pairs = [('caformer_s18', 'spanetv2_s18_hybrid'), ('spanetv2_s18_hybrid', 'caformer_s18'),
             ('convformer_s18', 'spanetv2_s18_pure'), ('spanetv2_s18_pure', 'convformer_s18')]
    res = {}
    models = {}
    for src, tgt in pairs:
        if tgt not in models:
            models[tgt] = C.Classifier(tgt).to(dev)
        for v in ('high', 'full'):
            D = np.load(os.path.join(FC.CACHE, 'deltas', f'{src}_{v}_eps1.0_s100.npy'), mmap_mode='r')
            corr = []
            with torch.no_grad():
                for i in range(0, len(X), BS):
                    x = C.to01(X[i:i + BS], dev)
                    d = torch.from_numpy(np.array(D[i:i + BS])).float().to(dev)
                    corr.append((models[tgt]((x + d).clamp(0, 1)).argmax(1).cpu().numpy() == Y[i:i + BS]))
            corr = np.concatenate(corr)
            res[f'{src}->{tgt}|{v}'] = dict(top1=float(corr.mean() * 100))
            print(src, '->', tgt, v, f'{corr.mean() * 100:.2f}', flush=True)
    FC.save_json('a_transfer.json', dict(eps=1.0, steps=100, n=len(X), results=res))


def load_images():
    idx = FC.subset1k()
    cache = C.ValCache(1.0)
    return np.array(cache.x[idx]), cache.y[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('part', choices=['main', 'large', 'apgd', 'apgd2', 'transfer'])
    ap.add_argument('--model', choices=MODELS)
    ap.add_argument('--variant', choices=['high', 'full'])
    args = ap.parse_args()
    C.set_fp32()
    # deterministic kernels: PGD outcomes are otherwise not reproducible. Exception: DeiT-S, whose patch-embedding
    # convolution backward is ~10x slower in deterministic mode (469 vs 44 ms per 50 images on an RTX 4090).
    DET[0] = args.model != 'deit_small'
    torch.backends.cudnn.benchmark = not DET[0]
    torch.backends.cudnn.deterministic = DET[0]
    dev = torch.device('cuda')
    if args.part == 'transfer':
        return part_transfer(args, dev)
    model = C.Classifier(args.model).to(dev)
    for p in model.parameters():
        p.requires_grad_(False)
    crop = C.MODELS[args.model]['crop']
    idx = FC.subset1k()
    cache = C.ValCache(crop)
    X, Y = np.array(cache.x[idx]), cache.y[idx]
    with torch.no_grad():
        clean = np.concatenate([(model(C.to01(X[i:i + BS], dev)).argmax(1).cpu().numpy() == Y[i:i + BS])
                                for i in range(0, len(X), BS)])
    FC.save_json(f'a_clean_{args.model}.json', dict(model=args.model, n=len(X), top1=float(clean.mean() * 100)))
    {'main': part_main, 'large': part_large, 'apgd': part_apgd, 'apgd2': part_apgd2}[args.part](args, model, X, Y, dev)


if __name__ == '__main__':
    main()
