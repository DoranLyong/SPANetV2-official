#!/usr/bin/env python
"""Figures from data/raw (all text >= 16 pt).

    python make_figs.py            # writes figs/*.png and data/*.csv (long format: model, experiment, setting, value)
"""
import csv
import glob
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, 'data', 'raw')
FIGS = os.path.join(HERE, 'figs')
DATA = os.path.join(HERE, 'data')

INK, INK2, MUTED, GRID = '#0b0b0b', '#52514e', '#898781', '#e1e0d9'
# colour = model family (reference categorical slots 1-6 in fixed order), dash = variant inside the family
CLS = {
    'spanetv2_s18_pure': ('SPANetV2-S18-pure', '#2a78d6', '-', 'o'),
    'spanetv2_s18_hybrid': ('SPANetV2-S18-hybrid', '#2a78d6', '--', 's'),
    'convformer_s18': ('ConvFormer-S18', '#eb6834', '-', 'o'),
    'caformer_s18': ('CAFormer-S18', '#eb6834', '--', 's'),
    'dfformer_s18': ('DFFormer-S18', '#1baf7a', '-', 'o'),
    'cdfformer_s18': ('CDFFormer-S18', '#1baf7a', '--', 's'),
    'convnext_tiny': ('ConvNeXt-T', '#eda100', '-', '^'),
    'swin_tiny': ('Swin-T', '#e87ba4', '-', 'v'),
    'deit_small': ('DeiT-S', '#008300', '-', 'D'),
}
SEQ = LinearSegmentedColormap.from_list('blue_seq', ['#f7fbff', '#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b'])

plt.rcParams.update({
    'font.size': 16, 'axes.titlesize': 18, 'axes.labelsize': 16, 'xtick.labelsize': 16, 'ytick.labelsize': 16,
    'legend.fontsize': 16, 'figure.titlesize': 20, 'axes.edgecolor': '#c3c2b7', 'axes.labelcolor': INK2,
    'xtick.color': INK2, 'ytick.color': INK2, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8,
    'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 110, 'font.family': 'DejaVu Sans',
})


def jl(name):
    p = os.path.join(RAW, name)
    return json.load(open(p)) if os.path.exists(p) else None


def legend(fig, keys, ncol=5):
    h = [Line2D([], [], color=CLS[k][1], ls=CLS[k][2], marker=CLS[k][3], ms=8, lw=2.2) for k in keys]
    fig.legend(h, [CLS[k][0] for k in keys], loc='outside lower center', ncol=ncol, frameon=False)


ROWS = []


def row(model, exp, setting, value):
    ROWS.append(dict(model=model, experiment=exp, setting=setting, value=value))


# ------------------------------------------------------------------------------------------------ 0
def clean_rows():
    for k in list(CLS) + ['convnext_tiny_ema']:
        d = jl(f'clean_{k}.json')
        if d:
            name = CLS[k][0] if k in CLS else 'ConvNeXt-T (EMA file)'
            row(name, '0_clean', 'top1 val50k fp32', d['top1'])
            row(name, '0_clean', 'top1 subset5k fp32', d['subset5k_top1'])
            row(name, '0_clean', 'top1 val50k fp16-autocast', d['fp16_top1'])


# ------------------------------------------------------------------------------------------------ A
def band_curves():
    keys = [k for k in CLS if jl(f'band_{k}.json')]
    if not keys:
        return
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.8), constrained_layout=True)
    for k in keys:
        d = jl(f'band_{k}.json')['top1']
        low = sorted((int(s[3:]), v) for s, v in d.items() if s.startswith('low'))
        high = sorted((int(s[4:]), v) for s, v in d.items() if s.startswith('high'))
        for ax, pts in ((axes[0], low), (axes[1], high)):
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=CLS[k][1], ls=CLS[k][2], marker=CLS[k][3], ms=7,
                    lw=2.2)
        for S, v in low:
            row(CLS[k][0], 'A_band_filter', f'lowpass S={S} | top1', v)
        for S, v in high:
            row(CLS[k][0], 'A_band_filter', f'highpass S={S} | top1', v)
    axes[0].set(title='Low-pass filtered ImageNet val (50k)', xlabel='filter size S', ylabel='top-1 (%)')
    axes[1].set(title='High-pass filtered ImageNet val (50k)', xlabel='filter size S', ylabel='top-1 (%)')
    axes[0].set_xticks(range(20, 221, 40))
    axes[1].set_xticks(range(170, 221, 10))
    legend(fig, keys)
    fig.savefig(os.path.join(FIGS, 'band_filter_curves.png'), bbox_inches='tight')
    plt.close(fig)


def band_examples():
    import sys
    sys.path.insert(0, HERE)
    import torch
    import common as C
    x8 = np.load(os.path.join(HERE, 'cache', 'val_crop1.000.npy'), mmap_mode='r')
    sub = C.subset5k()
    idx = [int(sub[i]) for i in (0, 2000, 4000)]
    x = torch.from_numpy(np.array(x8[idx])).permute(0, 3, 1, 2).float() / 255
    import exp_cls as E
    cols = [('clean', None)] + [(f'low S={S}', C.lowpass_mask(S, metric=E.MASK_METRIC)) for S in E.BAND_LOW_S] + \
           [(f'high S={S}', C.highpass_mask(S, metric=E.MASK_METRIC)) for S in E.BAND_HIGH_S]
    fig, axes = plt.subplots(3, len(cols), figsize=(1.9 * len(cols), 6.9), constrained_layout=True)
    for j, (name, m) in enumerate(cols):
        y = x if m is None else C.apply_mask(x, m)
        for i in range(3):
            im = y[i].permute(1, 2, 0).numpy()
            if name.startswith('high'):                 # zero-mean images: shown as 0.5 + 2*y for visibility
                im = 0.5 + 2 * im
            axes[i, j].imshow(np.clip(im, 0, 1))
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            axes[i, j].grid(False)
            for s in axes[i, j].spines.values():
                s.set_visible(False)
        axes[0, j].set_title(name.replace(' ', '\n'), fontsize=16)
    fig.suptitle('Band-filtered examples (low-pass shown clipped to [0,1]; high-pass shown as 0.5 + 2·x, clipped)',
                 fontsize=18)
    fig.savefig(os.path.join(FIGS, 'band_filter_examples.png'), bbox_inches='tight')
    plt.close(fig)
    with open(os.path.join(DATA, 'band_filter_examples.json'), 'w') as f:
        json.dump(dict(val_indices=idx, files=[open(os.path.join(HERE, 'cache', 'val_files.txt')).read().split()[i]
                                               for i in idx]), f, indent=1)


# ------------------------------------------------------------------------------------------------ B
def radial_dense(grid):
    H = grid.shape[0] // 2
    u, v = np.meshgrid(np.arange(-H, H + 1), np.arange(-H, H + 1), indexing='ij')
    r = np.sqrt(u ** 2 + v ** 2)
    ring = np.rint(r).astype(int)
    rs = np.arange(0, H + 1)                            # complete rings only (r <= 31)
    return rs / 112, np.array([np.nanmean(grid[ring == k]) for k in rs]), r / 112


def fourier_figs():
    keys = [k for k in CLS if os.path.exists(os.path.join(RAW, f'fourier_{k}.npy'))]
    if not keys:
        return
    grids = {k: np.load(os.path.join(RAW, f'fourier_{k}.npy')) * 100 for k in keys}
    vmin = min(np.nanmin(g) for g in grids.values())      # one colour range for every model: global min .. max
    vmax = max(np.nanmax(g) for g in grids.values())
    n = len(keys)
    nc = 3
    nr = int(np.ceil(n / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(5.2 * nc + 1.5, 5.0 * nr + 0.6), constrained_layout=True, squeeze=False)
    H = next(iter(grids.values())).shape[0] // 2
    for i, k in enumerate(keys):
        ax = axes[i // nc, i % nc]
        im = ax.imshow(grids[k], cmap=SEQ, vmin=vmin, vmax=vmax, origin='lower', extent=[-H - .5, H + .5, -H - .5, H + .5],
                       interpolation='nearest')
        ax.set_title(CLS[k][0])
        ax.grid(False)
        ax.set_xticks([-28, 0, 28])
        ax.set_yticks([-28, 0, 28])
    for j in range(n, nr * nc):
        axes[j // nc, j % nc].axis('off')
    cb = fig.colorbar(im, ax=axes, shrink=0.8, aspect=40)
    cb.set_label('error rate (%)')
    fig.suptitle('Fourier heat map (Yin et al.): error under Fourier-basis noise, l2 = 15.7 per channel, 5k val images\n'
                 f'axes: frequency offset from DC (cycles / 224 px), 63×63 centre; one colour range for all models '
                 f'({vmin:.1f}–{vmax:.1f} %)')
    fig.savefig(os.path.join(FIGS, 'fourier_heatmaps.png'), bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
    for k in keys:
        rho, e, rcell = radial_dense(grids[k])
        ax.plot(rho, e, color=CLS[k][1], ls=CLS[k][2], lw=2.2)
        sp = jl(f'fourier_sparse_{k}.json')
        lo = float(np.nanmean(grids[k][rcell <= 0.25 + 1e-9]))
        row(CLS[k][0], 'B_fourier', 'low band rho<=0.25 (63x63 window) | mean error %', lo)
        row(CLS[k][0], 'B_fourier', 'whole 63x63 window | mean error %', float(np.nanmean(grids[k])))
        for rr, ee in zip(rho, e):
            row(CLS[k][0], 'B_fourier', f'ring rho={rr:.4f} (dense) | mean error %', float(ee))
        if sp:
            pts = [(np.hypot(*map(int, key.split(','))) / 112, v * 100) for key, v in sp['error'].items()]
            pr = np.array(pts)
            bins = np.arange(0.25, 1.45, 8 / 112)
            mid, val = [], []
            for a, b in zip(bins[:-1], bins[1:]):
                sel = (pr[:, 0] > a) & (pr[:, 0] <= b)
                if sel.any():
                    mid.append(pr[sel, 0].mean())
                    val.append(pr[sel, 1].mean())
            ax.plot(mid, val, color=CLS[k][1], ls=CLS[k][2], marker=CLS[k][3], ms=7, lw=1.6)
            hi = float(pr[pr[:, 0] > 0.5, 1].mean())
            md = float(pr[(pr[:, 0] > 0.25) & (pr[:, 0] <= 0.5), 1].mean())
            row(CLS[k][0], 'B_fourier', 'high band rho>0.5 (stride-8 grid) | mean error %', hi)
            row(CLS[k][0], 'B_fourier', 'mid band 0.25<rho<=0.5 (stride-8 grid) | mean error %', md)
    for x in (0.25, 0.5):
        ax.axvline(x, color=MUTED, ls=':', lw=1.2)
    ax.set(xlabel='ρ = |frequency| / 112 (axis Nyquist = 1)', ylabel='error rate (%)',
           title='Fourier heat map, radial mean: lines = 63×63 window rings, markers = stride-8 grid beyond it')
    legend(fig, keys)
    fig.savefig(os.path.join(FIGS, 'fourier_radial.png'), bbox_inches='tight')
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ C
def pgd_fig():
    keys = [k for k in CLS if jl(f'pgd_{k}.json')]
    if not keys:
        return
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.8), constrained_layout=True, sharey=True)
    titles = {'full': '(i) unrestricted', 'low': '(ii) low band only (M_l^56)', 'high': '(iii) high band only (1 − M_l^56)'}
    for k in keys:
        d = jl(f'pgd_{k}.json')
        row(CLS[k][0], 'C_pgd', 'clean subset5k | top1', d['clean_top1'])
        for j, v in enumerate(('full', 'low', 'high')):
            eps = [0.0] + [float(e.split('|')[1]) for e in d['results'] if e.startswith(v + '|')]
            acc = [d['clean_top1']] + [r['top1'] for e, r in d['results'].items() if e.startswith(v + '|')]
            axes[j].plot(eps, acc, color=CLS[k][1], ls=CLS[k][2], marker=CLS[k][3], ms=8, lw=2.2)
            for e, r in d['results'].items():
                if e.startswith(v + '|'):
                    row(CLS[k][0], 'C_pgd', f'{v} eps={e.split("|")[1]} | top1', r['top1'])
                    if r.get('out_of_band_energy_mean') is not None:
                        row(CLS[k][0], 'C_pgd', f'{v} eps={e.split("|")[1]} | out-of-band energy fraction (mean)',
                            r['out_of_band_energy_mean'])
    for j, v in enumerate(('full', 'low', 'high')):
        axes[j].set(title=titles[v], xlabel='ε (L2, [0,1] pixels, 224×224×3)')
        axes[j].set_xticks([0, 0.25, 0.5, 1.0])
    axes[0].set_ylabel('top-1 (%) on 5k subset')
    legend(fig, keys)
    fig.savefig(os.path.join(FIGS, 'pgd_band.png'), bbox_inches='tight')
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ D
def dense_rows():
    for p in sorted(glob.glob(os.path.join(RAW, 'seg_*.json'))):
        d = json.load(open(p))
        b = jl(f'boundary_{d["model"]}.json')
        tag = f'D_{d["setting"]}'
        row(d['name'], tag, 'mIoU (single scale, this run)', d['mIoU'])
        if b:
            row(d['name'], tag, 'mIoU (recomputed from saved predictions)', b['mIoU'])
            row(d['name'], tag, 'Boundary IoU (d=2% diag, per-class mean)', b['boundary_IoU'])
    for p in sorted(glob.glob(os.path.join(RAW, 'det_*.json'))):
        d = json.load(open(p))
        b = jl(f'boundary_ap_{d["model"]}.json')
        tag = f'D3_{d["framework"]}'
        row(d['name'], tag, 'box AP (mmdet CocoMetric)', d['metrics'].get('bbox_mAP', float('nan')) * 100)
        row(d['name'], tag, 'mask AP (mmdet CocoMetric)', d['metrics'].get('segm_mAP', float('nan')) * 100)
        if b:
            row(d['name'], tag, 'mask AP (boundary-iou-api segm)', b['segm_same_api']['AP'])
            row(d['name'], tag, 'Boundary AP (d=0.02)', b['boundary']['AP'])


def boundary_examples():
    import sys
    import cv2
    sys.path.insert(0, HERE)
    import boundary_eval as BE
    ex = json.load(open(os.path.join(DATA, 'boundary_examples.json')))['images']
    order = ['upernet_spanetv2_s18_pure', 'upernet_spanetv2_s18_hybrid', 'upernet_convnext_tiny', 'upernet_swin_tiny',
             'upernet_spanetv2_s36_pure', 'upernet_spanetv2_s36_hybrid', 'upernet_convnext_small', 'upernet_swin_small',
             'fpn_spanetv2_s18_pure', 'fpn_spanetv2_s18_hybrid', 'fpn_poolformer_s24',
             'fpn_spanetv2_s36_pure', 'fpn_spanetv2_s36_hybrid', 'fpn_poolformer_s36']
    models = [m for m in order if os.path.isdir(os.path.join(HERE, 'cache', 'preds', 'seg', m))]
    if not models:
        return
    names = {}
    for m in models:
        d = jl(f'seg_{m}.json')
        names[m] = (('UperNet ' if m.startswith('upernet') else 'FPN ') + (d['name'] if d else m))
    ncol = 2 + len(models)
    fig, axes = plt.subplots(len(ex), ncol, figsize=(2.6 * ncol, 2.4 * len(ex) + 1.6), constrained_layout=True,
                             squeeze=False)
    rng = np.random.default_rng(0)
    pal = (rng.random((256, 3)) * 0.8 + 0.1)
    pal[255] = 0
    for i, stem in enumerate(ex):
        img = cv2.cvtColor(cv2.imread(os.path.join(BE.ADE, 'images', 'validation', stem + '.jpg')), cv2.COLOR_BGR2RGB)
        gt = BE.load_gt(stem)
        valid = gt != 255
        band = np.zeros_like(valid)
        for c in np.unique(gt[valid]):
            band |= BE.mask_to_boundary((gt == c).astype(np.uint8), BE.DIL).astype(bool)
        band &= valid
        gray = (img.mean(2, keepdims=True) / 255 * 0.5 + 0.5).repeat(3, 2)
        axes[i, 0].imshow(img)
        axes[i, 1].imshow(pal[np.where(valid, gt, 255)])
        for j, m in enumerate(models):
            pred = cv2.imread(os.path.join(HERE, 'cache', 'preds', 'seg', m, stem + '.png'), cv2.IMREAD_UNCHANGED)
            vis = gray.copy()
            vis[band & (pred == gt)] = (0.525, 0.714, 0.937)     # #86b6ef
            vis[band & (pred != gt)] = (0.816, 0.231, 0.231)     # #d03b3b
            axes[i, 2 + j].imshow(vis)
            err = (band & (pred != gt)).sum() / max(band.sum(), 1) * 100
            axes[i, 2 + j].text(0.02, 0.03, f'{err:.0f}%', transform=axes[i, 2 + j].transAxes, fontsize=16, color=INK,
                                bbox=dict(facecolor='white', alpha=0.8, lw=0, pad=1.5))
        for a in axes[i]:
            a.set_xticks([])
            a.set_yticks([])
            a.grid(False)
        axes[i, 0].set_ylabel(stem.replace('ADE_val_', ''), fontsize=16)
    for j, t in enumerate(['image', 'ground truth'] + [names[m] for m in models]):
        axes[0, j].set_title(t.replace(' ', '\n', 1) if j >= 2 else t, fontsize=16)
    fig.suptitle('Boundary errors on 6 thin-structure ADE20K val images: inside the ground-truth boundary band '
                 '(d = 2% of the diagonal), red = wrong class, blue = correct; number = % of band pixels wrong',
                 fontsize=18)
    fig.savefig(os.path.join(FIGS, 'boundary_examples.png'), bbox_inches='tight')
    plt.close(fig)


def write_csv():
    by = {}
    for r in ROWS:
        by.setdefault(r['experiment'].split('_')[0], []).append(r)
    with open(os.path.join(DATA, 'summary.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['model', 'experiment', 'setting', 'value'])
        w.writeheader()
        for r in ROWS:
            w.writerow({**r, 'value': f'{r["value"]:.6g}' if isinstance(r['value'], float) else r['value']})


def main():
    os.makedirs(FIGS, exist_ok=True)
    clean_rows()
    band_curves()
    band_examples()
    fourier_figs()
    pgd_fig()
    dense_rows()
    if os.path.exists(os.path.join(DATA, 'boundary_examples.json')):
        boundary_examples()
    write_csv()
    print(len(ROWS), 'rows')


if __name__ == '__main__':
    main()
