#!/usr/bin/env python
"""Figures from data/ (analyze.py). All text >= 16 pt.

    python make_figs.py
Main figures are for SPANetV2-S18-pure; '_R512' = 512x512 input, '_hybrid' = SPANetV2-S18-hybrid (stages 1-2 only).
"""
import csv
import gzip
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')
FIGS = os.path.join(HERE, 'figs')
INK, INK2, MUTED, GRID = '#0b0b0b', '#52514e', '#898781', '#e1e0d9'
BLUE, ORANGE = '#2a78d6', '#eb6834'
CK = {'IN1K': ('IN1K', INK, '-'), 'COCO/MaskRCNN': ('COCO Mask R-CNN', BLUE, '-'),
      'COCO/Cascade': ('COCO Cascade', BLUE, '--'), 'ADE20K/FPN': ('ADE20K FPN', ORANGE, '-'),
      'ADE20K/UperNet': ('ADE20K UperNet', ORANGE, '--')}
TR_TITLE = {'conv': 'conv  x1→x2', 'srf': 'SRF  x2→x3', 'exp': 'Exp share  x3→x4',
            'proj': 'Norm·act·proj  Σx4→s_ctx', 'mod_s': 'modulation  s_ctx→m', 'mod_q': 'modulation  q→m'}
XL = 'ρ   (0: low frequency, coarse shape  →  1: high frequency, fine detail)'

plt.rcParams.update({
    'font.size': 16, 'axes.titlesize': 17, 'axes.labelsize': 16, 'xtick.labelsize': 16, 'ytick.labelsize': 16,
    'legend.fontsize': 16, 'figure.titlesize': 19, 'axes.edgecolor': '#c3c2b7', 'axes.labelcolor': INK2,
    'xtick.color': INK2, 'ytick.color': INK2, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8,
    'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 100, 'font.family': 'DejaVu Sans',
})


def read(name, keycols):
    out = {}
    op = gzip.open if name.endswith('.gz') else open
    with op(os.path.join(DATA, name), 'rt') as f:
        for r in csv.DictReader(f):
            k = tuple(r[c] for c in keycols)
            out.setdefault(k, []).append((float(r['rho']), float(r['mean']), float(r['p25']), float(r['p75'])))
    return {k: np.array(sorted(v)) for k, v in out.items()}


def legend(fig):
    h = [Line2D([], [], color=c, ls=ls, lw=2.6) for _, c, ls in CK.values()]
    fig.legend(h, [n for n, _, _ in CK.values()], loc='outside lower center', ncol=5, frameon=False)


def plot_lines(ax, series):
    for c, (name, col, ls) in CK.items():
        a = series.get(c)
        if a is None:
            continue
        ax.fill_between(a[:, 0], a[:, 2], a[:, 3], color=col, alpha=0.10, lw=0)
        ax.plot(a[:, 0], a[:, 1], color=col, ls=ls, lw=2.2)
    for x in (0.25, 0.5, 1.0):
        ax.axvline(x, color=MUTED, ls=':', lw=1.1)


def gains(G, arch, R, suffix):
    stages = sorted({int(k[4]) for k in G if k[0] == arch and k[2] == str(R)})
    for s in stages:
        fig = plt.figure(figsize=(22, 25), constrained_layout=True)
        gs = fig.add_gridspec(6, 4)
        for row, tr in enumerate(['conv', 'srf', 'exp', 'proj', 'mod_s', 'mod_q']):
            heads = ['k=3', 'k=5', 'k=7', 'k=9'] if row < 3 else ['']
            for j, h in enumerate(heads):
                ax = fig.add_subplot(gs[row, j] if row < 3 else gs[row, :])
                ser = {c: G.get((arch, c, str(R), 'all', str(s), tr, h)) for c in CK}
                plot_lines(ax, ser)
                ax.axhline(0, color='#c3c2b7', lw=1.2)
                ax.set_title(f'{TR_TITLE[tr]}' + (f'   head {h}' if h else ''))
                if j == 0:
                    ax.set_ylabel('G = log10(P_out / P_in)')
                if row == 5:
                    ax.set_xlabel(XL)
        fig.suptitle(f'SPANetV2-S18-{arch.split("_")[-1]}, stage {s}, input {R}×{R}: transfer gain per SAG step '
                     f'(line = mean over 26 images, band = 25–75 %)', fontsize=19)
        legend(fig)
        fig.savefig(os.path.join(FIGS, f'transfer_gain_stage{s}{suffix}.png'), bbox_inches='tight')
        plt.close(fig)


def sctx(Pf, arch, name):
    stages = sorted({int(k[4]) for k in Pf if k[0] == arch})
    fig, axes = plt.subplots(2, len(stages), figsize=(5.6 * len(stages) + 1, 11), constrained_layout=True, squeeze=False)
    for i, R in enumerate((224, 512)):
        for j, s in enumerate(stages):
            ax = axes[i, j]
            plot_lines(ax, {c: Pf.get((arch, c, str(R), 'all', str(s), 's_ctx')) for c in CK})
            ax.set_title(f'stage {s}, input {R}×{R}')
            if j == 0:
                ax.set_ylabel('s_ctx: log10(P / P_DC)')
            if i == 1:
                ax.set_xlabel('ρ\n0: low freq. (coarse shape)\n1: high freq. (fine detail)')
    fig.suptitle(f'SPANetV2-S18-{arch.split("_")[-1]}: context s_ctx spectrum per backbone '
                 f'(line = mean over 26 images, band = 25–75 %)', fontsize=19)
    legend(fig)
    fig.savefig(os.path.join(FIGS, name), bbox_inches='tight')
    plt.close(fig)


def norm01(m):
    lo, hi = np.percentile(m, 1), np.percentile(m, 99)
    return np.clip((m - lo) / max(hi - lo, 1e-12), 0, 1)


def overlay(arch, R, stage, rows, name, title):
    imgs = np.load(os.path.join(DATA, f'images_R{R}.npy'))
    z = np.load(os.path.join(DATA, f'stage_maps_R{R}_{arch}.npz'))
    meta = json.load(open(os.path.join(DATA, 'images.json')))['images']
    fig, axes = plt.subplots(len(rows), 6, figsize=(16.5, 2.75 * len(rows) + 1.4), constrained_layout=True, squeeze=False)
    for r, i in enumerate(rows):
        axes[r, 0].imshow(imgs[i])
        axes[r, 0].set_ylabel(f"{meta[i]['group']}\n{meta[i]['file'].split('/')[-1].rsplit('.', 1)[0][-12:]}", fontsize=16)
        for j, c in enumerate(CK):
            axes[r, 1 + j].imshow(imgs[i])
            axes[r, 1 + j].imshow(norm01(z[f'{c}|s{stage}'][i]), cmap='jet', alpha=0.5, vmin=0, vmax=1)
        for a in axes[r]:
            a.set_xticks([])
            a.set_yticks([])
            a.grid(False)
    for j, t in enumerate(['input'] + [n for n, _, _ in CK.values()]):
        axes[0, j].set_title(t, fontsize=16)
    fig.suptitle(title, fontsize=18)
    if name.startswith('appendix'):          # appendix overlays: JPEG (quality 85, 80 dpi) to keep the folder small
        fig.savefig(os.path.join(FIGS, name.replace('.png', '.jpg')), bbox_inches='tight', dpi=80,
                    pil_kwargs=dict(quality=85))
    else:
        fig.savefig(os.path.join(FIGS, name), bbox_inches='tight')
    plt.close(fig)


def main():
    os.makedirs(FIGS, exist_ok=True)
    G = read('transfer_gain.csv.gz', ['arch', 'ckpt', 'res', 'group', 'stage', 'transition', 'head'])
    Pf = read('profiles.csv.gz', ['arch', 'ckpt', 'res', 'group', 'stage', 'tensor'])
    meta = json.load(open(os.path.join(DATA, 'images.json')))['images']
    main_rows = [i for i, m in enumerate(meta) if m['figure'] == 'main']
    appx_rows = [i for i, m in enumerate(meta) if m['figure'] == 'appendix']
    for arch, sfx in (('spanetv2_s18_pure', ''), ('spanetv2_s18_hybrid', '_hybrid')):
        if not any(k[0] == arch for k in G):
            continue
        gains(G, arch, 224, sfx)
        gains(G, arch, 512, f'_R512{sfx}')
        sctx(Pf, arch, f'sctx_spectra{sfx}.png')
        stages = (1, 2, 3, 4) if arch.endswith('pure') else (1, 2)
        for R in (224, 512):
            rs = '' if R == 224 else '_R512'
            for s in stages:
                t = (f'SPANetV2-S18-{arch.split("_")[-1]} context map, stage {s}, input {R}×{R}: mean |s_ctx| over '
                     f'channels and blocks, per image 1–99 % → jet, alpha 0.5')
                main_name = (f'context_overlay_stage{s}{rs}{sfx}.png' if (s == stages[-1] and R == 224)
                             else f'appendix_context_overlay_stage{s}{rs}{sfx}.png')
                overlay(arch, R, s, main_rows, main_name, t)
                overlay(arch, R, s, appx_rows[:9], f'appendix_context_overlay_stage{s}{rs}{sfx}_images_a.png',
                        t + ' (appendix images 1–9)')
                overlay(arch, R, s, appx_rows[9:], f'appendix_context_overlay_stage{s}{rs}{sfx}_images_b.png',
                        t + ' (appendix images 10–17)')
    print('figures written')


if __name__ == '__main__':
    main()
