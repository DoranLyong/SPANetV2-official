"""Figures for analyze_srf.py (called at the end of `analyze`; can be re-run alone:
    python plot_srf.py S18pure
). All text >= 16 pt."""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

KSIZE = {1: 3, 2: 5, 3: 7, 4: 9}
INK, INK2, MUTED, GRID = '#0b0b0b', '#52514e', '#898781', '#e1e0d9'
BLUE, ORANGE = '#2a78d6', '#eb6834'          # categorical slots 1-2 of the reference palette
STYLE = {                                     # colour = dataset, dash = method inside the dataset
    'IN1K': dict(color=INK, ls='-', name='IN1K'),
    'COCO/MaskRCNN': dict(color=BLUE, ls='-', name='COCO (Mask R-CNN)'),
    'COCO/Cascade': dict(color=BLUE, ls='--', name='COCO (Cascade)'),
    'ADE20K/FPN': dict(color=ORANGE, ls='-', name='ADE20K (FPN)'),
    'ADE20K/UperNet': dict(color=ORANGE, ls='--', name='ADE20K (UperNet)'),
}
SEQ = LinearSegmentedColormap.from_list('blue_seq', ['#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b'])
DIV = LinearSegmentedColormap.from_list('blue_gray_red', ['#104281', '#2a78d6', '#f0efec', '#e34948', '#8f1d1d'])

plt.rcParams.update({
    'font.size': 16, 'axes.titlesize': 18, 'axes.labelsize': 16, 'xtick.labelsize': 16, 'ytick.labelsize': 16,
    'legend.fontsize': 16, 'figure.titlesize': 20, 'axes.edgecolor': '#c3c2b7', 'axes.labelcolor': INK2,
    'xtick.color': INK2, 'ytick.color': INK2, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8,
    'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 110, 'font.family': 'DejaVu Sans',
})


def _load(out_dir):
    with open(os.path.join(out_dir, 'inspect.json')) as f:
        ins = json.load(f)
    rad = dict(np.load(os.path.join(out_dir, 'srf_radial.npz')))
    maps = dict(np.load(os.path.join(out_dir, 'psi_maps.npz')))
    return ins, rad, maps


def _bands(ax):
    for x in (0.25, 0.5):
        ax.axvline(x, color=MUTED, lw=1.0, ls=':', zorder=0)


def _legend(fig, labels, extra=None):
    handles = [Line2D([], [], color=STYLE[l]['color'], ls=STYLE[l]['ls'], lw=2.5) for l in labels]
    names = [STYLE[l]['name'] for l in labels]
    if extra:
        handles += extra[0]
        names += extra[1]
    fig.legend(handles, names, loc='outside lower center', ncol=min(len(names), 6), frameon=False)


def radial_grid(model, ins, rad, labels, quantity, ylabel, fname, figs, zero_line=False, iqr_label='IN1K',
                title=''):
    stages, heads = ins['stages'], ins['heads']
    fig, axes = plt.subplots(len(stages), len(heads), figsize=(4.6 * len(heads), 3.9 * len(stages) + 1.2),
                             squeeze=False, constrained_layout=True)
    for i, s in enumerate(stages):
        rho = rad[f'rho|{model}|s{s}']
        mk = 'o' if len(rho) <= 10 else None
        for j, h in enumerate(heads):
            ax = axes[i, j]
            k = KSIZE[h]
            _bands(ax)
            if zero_line:
                ax.axhline(0, color='#c3c2b7', lw=1.2, zorder=0)
            for l in labels:
                key = f'{model}|{l}|s{s}|k{k}|{quantity}'
                if key not in rad:
                    continue
                x = rad[key]
                m = np.nanmean(x, 0)
                if l == iqr_label:
                    ax.fill_between(rho, np.nanpercentile(x, 25, 0), np.nanpercentile(x, 75, 0),
                                    color=STYLE[l]['color'], alpha=0.12, lw=0, zorder=1)
                ax.plot(rho, m, color=STYLE[l]['color'], ls=STYLE[l]['ls'], lw=2.2, marker=mk, ms=7, zorder=3)
            ax.set_title(f'stage {s} · k={k}  ({ins["grid"][str(s)]["H"]}×{ins["grid"][str(s)]["Wr"]})', pad=6)
            ax.set_xlim(-0.03, 1.45)
            if i == len(stages) - 1:
                ax.set_xlabel('ρ  (axis Nyquist = 1)')
            if j == 0:
                ax.set_ylabel(ylabel)
    extra = None
    if iqr_label in labels:
        extra = ([plt.Rectangle((0, 0), 1, 1, color=STYLE[iqr_label]['color'], alpha=0.12)],
                 [f'{STYLE[iqr_label]["name"]} 25–75%'])
    if title:
        fig.suptitle(title, color=INK, fontsize=18)
    _legend(fig, labels, extra)
    fig.savefig(os.path.join(figs, fname), bbox_inches='tight')
    plt.close(fig)


def psi_map(model, ins, maps, labels, figs):
    stages = ins['stages']
    tasks = [l for l in labels if l != 'IN1K']
    rows = [(l, 'psi') for l in labels] + [(l, 'diff') for l in tasks]
    fig, axes = plt.subplots(len(rows), len(stages), figsize=(3.9 * len(stages) + 2.2, 3.3 * len(rows)),
                             squeeze=False, constrained_layout=True)
    for j, s in enumerate(stages):
        ref = maps[f'map|{model}|IN1K|s{s}']
        H = ref.shape[0]
        f = np.fft.fftshift(np.fft.fftfreq(H)) / 0.5
        d = f[1] - f[0] if H > 1 else 1
        ext = [f[0] - d / 2, f[-1] + d / 2, f[0] - d / 2, f[-1] + d / 2]
        psis = [maps[f'map|{model}|{l}|s{s}'] for l in labels]
        vmin, vmax = min(p.min() for p in psis), max(p.max() for p in psis)
        diffs = [maps[f'map|{model}|{l}|s{s}'] - ref for l in tasks]
        vd = max(np.abs(x).max() for x in diffs) if diffs else 1
        im_p = im_d = None
        ax_p, ax_d = [], []
        for i, (l, kind) in enumerate(rows):
            ax = axes[i, j]
            ax.grid(False)
            if kind == 'psi':
                im_p = ax.imshow(maps[f'map|{model}|{l}|s{s}'], cmap=SEQ, vmin=vmin, vmax=vmax, origin='lower',
                                 extent=ext, interpolation='nearest')
                ax_p.append(ax)
            else:
                im_d = ax.imshow(maps[f'map|{model}|{l}|s{s}'] - ref, cmap=DIV, vmin=-vd, vmax=vd, origin='lower',
                                 extent=ext, interpolation='nearest')
                ax_d.append(ax)
            ax.set_xticks([-1, 0, 1])
            ax.set_yticks([-1, 0, 1])
            if i == 0:
                ax.set_title(f'stage {s}  ({H}×{H})', pad=8)
            if i == len(rows) - 1:
                ax.set_xlabel('kx / 0.5')
            if j == 0:
                nm = STYLE[l]['name'].replace(' (', '\n(')
                ax.set_ylabel(f'{nm}\nψ' if kind == 'psi' else f'{nm}\nψ − IN1K', fontsize=16)
        cb = fig.colorbar(im_p, ax=ax_p, shrink=0.9, aspect=30, pad=0.02)
        cb.ax.tick_params(labelsize=16)
        cb.set_label('ψ', fontsize=16)
        if ax_d:
            cb = fig.colorbar(im_d, ax=ax_d, shrink=0.9, aspect=30, pad=0.02)
            cb.ax.tick_params(labelsize=16)
            cb.set_label('Δψ', fontsize=16)
            cb.formatter.set_powerlimits((-2, 2))
            cb.update_ticks()
    fig.suptitle(f'SPANetV2-{model}: ψ averaged over channels, heads and blocks (fftshift, DC at centre)',
                 color=INK, fontsize=18)
    fig.savefig(os.path.join(figs, f'psi_map_{model}.png'), bbox_inches='tight')
    plt.close(fig)


def plot_model(model, out_dir, figs):
    ins, rad, maps = _load(out_dir)
    labels = ins['labels']
    tasks = [l for l in labels if l != 'IN1K']
    radial_grid(model, ins, rad, labels, 'psi', 'ψ = σ(M)', f'psi_radial_{model}.png', figs,
                title=f'SPANetV2-{model}: ψ ring mean over (block, channel)')
    radial_grid(model, ins, rad, tasks, 'delta_psi', 'Δψ = ψ_task − ψ_IN1K', f'dpsi_radial_{model}.png', figs,
                zero_line=True, iqr_label=None, title=f'SPANetV2-{model}: Δψ ring mean over (block, channel)')
    radial_grid(model, ins, rad, labels, 'effective_norm', 'ψ·|K̂| / channel mean', f'effective_radial_{model}.png',
                figs, title=f'SPANetV2-{model}: effective response ψ·|K̂|, each channel divided by its mean')
    radial_grid(model, ins, rad, labels, 'kernel_norm', '|K̂| / channel mean', f'kernel_radial_{model}.png', figs,
                title=f'SPANetV2-{model}: depthwise kernel |K̂|, each channel divided by its mean')
    psi_map(model, ins, maps, labels, figs)


if __name__ == '__main__':
    here = os.path.dirname(os.path.abspath(__file__))
    m = sys.argv[1]
    plot_model(m, os.path.join(here, 'data', 'per_model', m), os.path.join(here, 'figs'))
