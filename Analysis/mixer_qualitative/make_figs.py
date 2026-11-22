#!/usr/bin/env python
"""Figures and CSVs for E1 / E2 from data/ (extract.py). All text >= 16 pt; one colour range per figure.

    python make_figs.py
"""
import csv
import importlib.util
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
AN = os.path.dirname(HERE)
DATA = os.path.join(HERE, 'data')
FIGS = os.path.join(HERE, 'figs')
_spec = importlib.util.spec_from_file_location('_srf', os.path.join(AN, 'srf_task_analysis', 'analyze_srf.py'))
SRF = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(SRF)

INK, INK2, MUTED, GRID = '#0b0b0b', '#52514e', '#898781', '#e1e0d9'
SEQ = LinearSegmentedColormap.from_list('blue_seq', ['#f7fbff', '#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b'])
DIV = LinearSegmentedColormap.from_list('div', ['#104281', '#2a78d6', '#f0efec', '#e34948', '#8f1d1d'])
LOWHI = 'ρ   (0 = low frequency, coarse shape  →  1 = high frequency, fine detail)'
MAPNOTE = 'centre = low frequency (coarse shape), edges = high frequency (fine detail)'
STYLE = {   # branch -> (label, colour, dash)
    'ConvFormer|dwconv': ('ConvFormer: depthwise conv 7×7', '#eb6834', '-'),
    'ConvFormer|mixer': ('ConvFormer: mixer output', '#eb6834', '--'),
    'DFFormer|dynamic filter': ('DFFormer: after dynamic filter', '#1baf7a', '-'),
    'DFFormer|mixer': ('DFFormer: mixer output', '#1baf7a', '--'),
    'SPANetV2|head k=3': ('SPANetV2: head k=3 (SRF out)', '#86b6ef', '-'),
    'SPANetV2|head k=5': ('SPANetV2: head k=5 (SRF out)', '#3987e5', '-'),
    'SPANetV2|head k=7': ('SPANetV2: head k=7 (SRF out)', '#1c5cab', '-'),
    'SPANetV2|head k=9': ('SPANetV2: head k=9 (SRF out)', '#0d366b', '-'),
    'SPANetV2|mixer': ('SPANetV2: mixer output', INK, '--'),
    'CAFormer|attention': ('CAFormer: attention output', '#e87ba4', '-'),
}
MAP_COLS = ['ConvFormer|dwconv', 'DFFormer|dynamic filter', 'SPANetV2|head k=3', 'SPANetV2|head k=5',
            'SPANetV2|head k=7', 'SPANetV2|head k=9', 'SPANetV2|mixer', 'CAFormer|attention']
MAP_TITLE = {'ConvFormer|dwconv': 'ConvFormer\ndepthwise conv', 'DFFormer|dynamic filter': 'DFFormer\ndynamic filter',
             'SPANetV2|head k=3': 'SPANetV2\nhead k=3', 'SPANetV2|head k=5': 'SPANetV2\nhead k=5',
             'SPANetV2|head k=7': 'SPANetV2\nhead k=7', 'SPANetV2|head k=9': 'SPANetV2\nhead k=9',
             'SPANetV2|mixer': 'SPANetV2\nmixer output', 'CAFormer|attention': 'CAFormer\nattention'}
DF_IMAGES = [0, 4, 8]            # first image of each group (ImageNet random, ImageNet Sobel, ADE20K)

plt.rcParams.update({
    'font.size': 16, 'axes.titlesize': 17, 'axes.labelsize': 16, 'xtick.labelsize': 16, 'ytick.labelsize': 16,
    'legend.fontsize': 16, 'figure.titlesize': 19, 'axes.edgecolor': '#c3c2b7', 'axes.labelcolor': INK2,
    'xtick.color': INK2, 'ytick.color': INK2, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8,
    'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 110, 'font.family': 'DejaVu Sans',
})


def clean_ax(a):
    a.set_xticks([])
    a.set_yticks([])
    a.grid(False)
    for s in a.spines.values():
        s.set_visible(False)


def norm01(m):
    lo, hi = np.percentile(m, 1), np.percentile(m, 99)
    return np.clip((m - lo) / max(hi - lo, 1e-12), 0, 1)


# ------------------------------------------------------------------------------------------------ E1
def e1_maps(imgs, meta):
    main = [i for i, m in enumerate(meta) if m['figure'] == 'main']
    appx = [i for i, m in enumerate(meta) if m['figure'] == 'appendix']
    for s in (1, 2, 3):
        z = np.load(os.path.join(DATA, f'e1_maps_stage{s}.npz'))
        cols = [c for c in MAP_COLS if c in z.files]
        for which, rows in (('', main), ('_appendix', appx)):
            fig, axes = plt.subplots(len(rows), 1 + len(cols), figsize=(2.55 * (1 + len(cols)), 2.6 * len(rows) + 1.5),
                                     constrained_layout=True, squeeze=False)
            for r, i in enumerate(rows):
                axes[r, 0].imshow(imgs[i])
                clean_ax(axes[r, 0])
                tag = meta[i]['file'].split('/')[-1].replace('ILSVRC2012_val_', 'IN ').replace('ADE_val_', 'ADE ')
                axes[r, 0].set_ylabel(tag.rsplit('.', 1)[0] + f"\n({meta[i]['group'].replace('_', ' ')})", fontsize=16)
                for j, c in enumerate(cols):
                    im = axes[r, 1 + j].imshow(norm01(z[c][i]), cmap=SEQ, vmin=0, vmax=1, interpolation='nearest')
                    clean_ax(axes[r, 1 + j])
            axes[0, 0].set_title('input')
            for j, c in enumerate(cols):
                axes[0, 1 + j].set_title(MAP_TITLE[c])
            cb = fig.colorbar(im, ax=axes, shrink=0.6, aspect=40)
            cb.set_label('channel L2 norm, per image 1–99 percentile → 0–1')
            H = z[cols[0]].shape[-1]
            fig.suptitle(f'Stage {s} ({H}×{H} tokens): where each token-mixer branch responds'
                         + (' — appendix images' if which else ''), fontsize=19)
            fig.savefig(os.path.join(FIGS, f'e1_maps_stage{s}{which}.png'), bbox_inches='tight')
            plt.close(fig)


def e1_spectra(rows):
    fig, axes = plt.subplots(1, 3, figsize=(24, 7.6), constrained_layout=True)
    seen = []
    for k, s in enumerate((1, 2, 3)):
        z = np.load(os.path.join(DATA, f'e1_spectra_stage{s}.npz'))
        rho = z['rho']
        ax = axes[k]
        for br in STYLE:
            if br not in z.files:
                continue
            y = z[br]
            m, p25, p75 = y.mean(0), np.percentile(y, 25, 0), np.percentile(y, 75, 0)
            lab, col, ls = STYLE[br]
            ax.fill_between(rho, p25, p75, color=col, alpha=0.10, lw=0)
            ax.plot(rho, m, color=col, ls=ls, lw=2.4)
            if br not in seen:
                seen.append(br)
            model, branch = br.split('|')
            for r, (a, b, c) in enumerate(zip(m, p25, p75)):
                rows.append(dict(model=model, stage=s, branch=branch, ring=r, rho=f'{rho[r]:.6g}', mean=f'{a:.6g}',
                                 p25=f'{b:.6g}', p75=f'{c:.6g}'))
        for x in (0.25, 0.5, 1.0):
            ax.axvline(x, color=MUTED, ls=':', lw=1.2)
        ax.set(title=f'Stage {s}', xlabel='ρ\n0: low frequency (coarse shape)\n1: high frequency (fine detail)',
               ylabel='log10( P / P_DC ),  ring mean' if k == 0 else '')
    h = [Line2D([], [], color=STYLE[b][1], ls=STYLE[b][2], lw=2.6) for b in seen]
    fig.legend(h, [STYLE[b][0] for b in seen], loc='outside lower center', ncol=4, frameon=False)
    fig.suptitle('Spectra of the token-mixer outputs on 12 images: channel-mean power relative to DC '
                 '(line = mean over images, band = 25–75 %)', fontsize=19)
    fig.savefig(os.path.join(FIGS, 'e1_spectra.png'), bbox_inches='tight')
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ E2
def per_channel_norm(g, A):
    return A / g.wmean(A)[:, None, None]


def e2(rows):
    gains = {}
    summary = {}
    for s in (1, 2, 3):
        cf = np.load(os.path.join(DATA, f'e2_params_convformer_s18_stage{s}.npz'))
        sp = np.load(os.path.join(DATA, f'e2_params_spanetv2_s18_pure_stage{s}.npz'))
        df = np.load(os.path.join(DATA, f'e2_params_dfformer_s18_stage{s}.npz'))
        H, Wr = sp['SPANetV2|k=3|M'].shape[1:]
        g = SRF.Grid(H, Wr)
        cols = []                                     # (title, branch, normalised array (N, H, Wr))
        cols.append(('ConvFormer\n|K̂| (7×7)', 'ConvFormer |K| dwconv', per_channel_norm(g, g.kernel_dtft(cf['ConvFormer|dwconv weight']))))
        for k in (3, 5, 7, 9):
            Kh = g.kernel_dtft(sp[f'SPANetV2|k={k}|kernel'])
            psi = SRF.sigmoid(sp[f'SPANetV2|k={k}|M'])
            gains[(s, k)] = psi
            summary[f'{s}|SPANetV2 k={k} psi'] = dict(stage=s, branch=f'SPANetV2 k={k} psi',
                                                    **{f'band_{b}' if b != 'DC' else 'DC': float(g.wmean(psi, g.band[b]).mean())
                                                       for b in ('DC', 'low', 'mid', 'high')})
            cols.append((f'SPANetV2 k={k}\nψ·|K̂|', f'SPANetV2 k={k} effective', per_channel_norm(g, psi * Kh)))
            for name, arr in ((f'SPANetV2 k={k} |K|', per_channel_norm(g, Kh)), (f'SPANetV2 k={k} psi', psi)):
                prof = g.radial(arr)
                for r in range(prof.shape[1]):
                    rows.append(dict(model=name.split()[0], stage=s, branch=' '.join(name.split()[1:]), ring=r,
                                     rho=f'{g.ring_rho[r]:.6g}', mean=f'{prof[:, r].mean():.6g}',
                                     p25=f'{np.percentile(prof[:, r], 25):.6g}', p75=f'{np.percentile(prof[:, r], 75):.6g}'))
        basis, routing = df['DFFormer|basis'], df['DFFormer|routing']
        for b in DF_IMAGES:
            Wb = np.abs(np.einsum('fc,hwf->chw', routing[b], basis))
            cols.append((f'DFFormer\nimage {b + 1}', f'DFFormer predicted |filter| image {b + 1}', per_channel_norm(g, Wb)))
        maps = [g.full_map(c[2].mean(0)) for c in cols]
        profs = [g.radial(c[2]) for c in cols]
        vmax = max(np.percentile(m, 99.5) for m in maps)
        fig, axes = plt.subplots(2, len(cols), figsize=(2.9 * len(cols), 6.8), constrained_layout=True,
                                 gridspec_kw=dict(height_ratios=[1.15, 1]))
        f = np.fft.fftshift(np.fft.fftfreq(H)) / 0.5
        d = f[1] - f[0]
        ext = [f[0] - d / 2, f[-1] + d / 2, f[0] - d / 2, f[-1] + d / 2]
        ymax = max(np.percentile(p, 75, 0).max() for p in profs) * 1.05
        for j, (title, branch, arr) in enumerate(cols):
            im = axes[0, j].imshow(maps[j], cmap=SEQ, vmin=0, vmax=vmax, origin='lower', extent=ext, interpolation='nearest')
            axes[0, j].set_title(title)
            axes[0, j].grid(False)
            axes[0, j].set_xticks([-1, 0, 1])
            axes[0, j].set_yticks([-1, 0, 1] if j == 0 else [])
            p = profs[j]
            ax = axes[1, j]
            ax.fill_between(g.ring_rho, np.percentile(p, 25, 0), np.percentile(p, 75, 0), color='#3987e5', alpha=0.2, lw=0)
            ax.plot(g.ring_rho, p.mean(0), color='#1c5cab', lw=2.4)
            ax.axhline(1, color=MUTED, lw=1, ls=':')
            ax.set_ylim(0, ymax)
            ax.set_xlim(0, 1.45)
            ax.set_xticks([0, 0.5, 1])
            if j:
                ax.set_yticklabels([])
            for r in range(p.shape[1]):
                rows.append(dict(model=branch.split()[0], stage=s, branch=' '.join(branch.split()[1:]), ring=r,
                                 rho=f'{g.ring_rho[r]:.6g}', mean=f'{p[:, r].mean():.6g}',
                                 p25=f'{np.percentile(p[:, r], 25):.6g}', p75=f'{np.percentile(p[:, r], 75):.6g}'))
            lo = p[:, (g.ring_rho > 0) & (g.ring_rho <= 0.25)].mean() if ((g.ring_rho > 0) & (g.ring_rho <= 0.25)).any() else np.nan
            summary[f'{s}|{branch}'] = dict(stage=s, branch=branch,
                                           band_low=float(g.wmean(arr, g.band['low']).mean()),
                                           band_mid=float(g.wmean(arr, g.band['mid']).mean()),
                                           band_high=float(g.wmean(arr, g.band['high']).mean()),
                                           DC=float(g.wmean(arr, g.band['DC']).mean()))
        axes[1, 0].set_ylabel('response / channel mean')
        axes[0, 0].set_ylabel('ky / Nyquist')
        fig.supxlabel(LOWHI, fontsize=16)
        cb = fig.colorbar(im, ax=axes[0, :], shrink=0.85, aspect=25)
        cb.set_label('response / channel mean')
        fig.suptitle(f'Stage {s} ({H}×{H}): filter magnitude responses (map: {MAPNOTE}; '
                     f'curve: ring mean, band = 25–75 % over channels)', fontsize=19)
        fig.savefig(os.path.join(FIGS, f'e2_filters_stage{s}.png'), bbox_inches='tight')
        plt.close(fig)
        if s == 1:
            np.savez_compressed(os.path.join(DATA, 'e2_dfformer_basis_and_routing.npz'),
                                note=np.array('per stage: basis (H, W//2+1, F) complex, routing (12 images, F, 2C)'),
                                **{f'stage{t}_{k}': np.load(os.path.join(DATA, f'e2_params_dfformer_s18_stage{t}.npz'))[f'DFFormer|{k}']
                                   for t in (1, 2, 3) for k in ('basis', 'routing')})
    # psi maps
    dev = max(abs(g_.mean(0) - 0.5).max() for g_ in gains.values())
    fig, axes = plt.subplots(3, 4, figsize=(17, 13.5), constrained_layout=True)
    for (s, k), psi in gains.items():
        H, Wr = psi.shape[1:]
        g = SRF.Grid(H, Wr)
        f = np.fft.fftshift(np.fft.fftfreq(H)) / 0.5
        d = f[1] - f[0]
        ext = [f[0] - d / 2, f[-1] + d / 2, f[0] - d / 2, f[-1] + d / 2]
        ax = axes[s - 1, (k - 3) // 2]
        im = ax.imshow(g.full_map(psi.mean(0)), cmap=DIV, vmin=0.5 - dev, vmax=0.5 + dev, origin='lower', extent=ext,
                       interpolation='nearest')
        ax.grid(False)
        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([-1, 0, 1])
        if s == 1:
            ax.set_title(f'head k={k}')
        if k == 3:
            ax.set_ylabel(f'stage {s} ({H}×{H})\nky / Nyquist')
    cb = fig.colorbar(im, ax=axes, shrink=0.7, aspect=40)
    cb.set_label('ψ = sigmoid(M), channel mean  (0.5 = initial value)')
    fig.suptitle(f'SPANetV2-S18-pure SRF gain ψ per head ({MAPNOTE};\n'
                 'red > 0.5 passes more than at initialisation, blue < 0.5 passes less)', fontsize=19)
    fig.savefig(os.path.join(FIGS, 'e2_srf_gain.png'), bbox_inches='tight')
    plt.close(fig)
    json.dump(summary, open(os.path.join(DATA, 'e2_band_summary.json'), 'w'), indent=1)


def write_csv(path, rows):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['model', 'stage', 'branch', 'ring', 'rho', 'mean', 'p25', 'p75'])
        w.writeheader()
        w.writerows(rows)


def main():
    os.makedirs(FIGS, exist_ok=True)
    imgs = np.load(os.path.join(DATA, 'images_u8.npy'))
    meta = json.load(open(os.path.join(DATA, 'images.json')))['images']
    e1_maps(imgs, meta)
    r1 = []
    e1_spectra(r1)
    write_csv(os.path.join(DATA, 'e1_radial.csv'), r1)
    r2 = []
    e2(r2)
    write_csv(os.path.join(DATA, 'e2_radial.csv'), r2)
    print('figures written', len(r1), len(r2))


if __name__ == '__main__':
    main()
