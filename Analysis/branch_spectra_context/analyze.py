#!/usr/bin/env python
"""Stage-level spectra, transfer gains, task contrasts, context-map correlations and object ratios from data/raw.

    python analyze.py

Definitions
  profile      ring mean of log10(P / P_DC) of a tensor, averaged over the SPAM blocks of the stage (per image)
  band value   cell mean of log10(P / P_DC) over a band (low 0 < rho <= 0.25, mid 0.25 < rho <= 0.5, high rho > 0.5)
  gain G       log10(P_out / P_in) with P = ring mean (profile) or band cell mean (band value) of the channel-mean
               power; computed per block and per image, then averaged over the blocks of the stage
               conv_h  x1_h -> x2_h | srf_h  x2_h -> x3_h | exp_h  x3_h -> x4_h | proj  sum_h x4_h -> s_ctx
               mod_s   s_ctx -> m   | mod_q  q -> m
  statistics   mean, p25, p75 over the images of a group ('all' = every image)
  contrast     per (arch, resolution, image group, stage, quantity, head, band): v = image mean per checkpoint;
               W = max(|MaskRCNN - Cascade|, |FPN - UperNet|); X = |COCO checkpoint - ADE20K checkpoint| for the
               4 cross pairs; a cell counts when min X > W (C-4 (e))
  context map  per stage: mean over blocks of the channel mean of |s_ctx| (= notebook cells 19-21: concatenate the
               blocks' |s_ctx| along channels and average), bilinear upsampling to the input size
Writes data/profiles.csv.gz, data/transfer_gain.csv.gz, data/band_values.csv, data/contrast.csv,
data/contrast_summary.csv, data/map_correlation.csv, data/map_correlation_summary.csv, data/object_ratio.csv,
data/stage_maps_R{224,512}_<arch>.npz.
"""
import csv
import gzip
import itertools
import json
import os
import re

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, 'data', 'raw')
CKPTS = ['IN1K', 'COCO/MaskRCNN', 'COCO/Cascade', 'ADE20K/FPN', 'ADE20K/UperNet']
COCO, ADE = ['COCO/MaskRCNN', 'COCO/Cascade'], ['ADE20K/FPN', 'ADE20K/UperNet']
RES = (224, 512)
BANDS = ('low', 'mid', 'high')
TRANS = [('conv', 'x1_h{h}', 'x2_h{h}'), ('srf', 'x2_h{h}', 'x3_h{h}'), ('exp', 'x3_h{h}', 'x4_h{h}')]
TRANS_ALL = [('proj', 'sum_x4', 's_ctx'), ('mod_s', 's_ctx', 'm'), ('mod_q', 'q', 'm')]
TENSORS = [f'x{i}_h{h}' for i in (1, 2, 3, 4) for h in range(4)] + ['sum_x4', 's_ctx', 'q', 'm']


def stats(a):
    return a.mean(0), np.percentile(a, 25, 0), np.percentile(a, 75, 0)


def load(arch):
    D = {}
    for c in CKPTS:
        for R in RES:
            p = os.path.join(RAW, arch, f'{c.replace("/", "_")}_R{R}.npz')
            if os.path.exists(p):
                D[(c, R)] = dict(np.load(p))
                D[(c, R)].update(np.load(p.replace('.npz', '_maps.npz')))
    return D


def blocks_of(z):
    out = {}
    for k in z:
        m = re.match(r's(\d+)b(\d+)\|s_ctx\|ringP$', k)
        if m:
            out.setdefault(int(m[1]), []).append(int(m[2]))
    return {s: sorted(v) for s, v in sorted(out.items())}


def main():
    meta = json.load(open(os.path.join(HERE, 'data', 'images.json')))['images']
    groups = {'all': np.arange(len(meta))}
    for g in ('imagenet', 'coco', 'ade20k'):
        groups[g] = np.array([i for i, m in enumerate(meta) if m['group'] == g])
    prof_rows, gain_rows, band_rows, contrast_rows = [], [], [], []
    corr_rows, obj_rows = [], []
    vals = {}           # (arch, R, group, stage, quantity, head, band) -> {ckpt: image mean}
    for arch in sorted(os.listdir(RAW)):
        D = load(arch)
        if not D:
            continue
        for R in RES:
            masks = np.load(os.path.join(HERE, 'data', f'masks_R{R}.npy'))
            stage_maps = {}
            for c in CKPTS:
                if (c, R) not in D:
                    continue
                z = D[(c, R)]
                BL = blocks_of(z)
                for s, bl in BL.items():
                    H = z[f'map|s{s}b{bl[0]}'].shape[-1]
                    rho = z[f'rho|{H}x{H}']
                    # ---- profiles
                    for t in TENSORS:
                        prof = np.mean([z[f's{s}b{b}|{t}|ringL'] for b in bl], 0)        # (N, R)
                        bandL = np.mean([z[f's{s}b{b}|{t}|bandL'] for b in bl], 0)      # (N, 3)
                        for gname, idx in groups.items():
                            m, p25, p75 = stats(prof[idx])
                            for r in range(len(rho)):
                                prof_rows.append((arch, c, R, gname, s, t, r, f'{rho[r]:.5g}', f'{m[r]:.5g}',
                                                  f'{p25[r]:.5g}', f'{p75[r]:.5g}'))
                            if t == 's_ctx':
                                bm, b25, b75 = stats(bandL[idx])
                                for j, bn in enumerate(BANDS):
                                    band_rows.append((arch, c, R, gname, s, 's_ctx log10(P/P_DC)', '', bn,
                                                      f'{bm[j]:.6g}', f'{b25[j]:.6g}', f'{b75[j]:.6g}'))
                                    vals.setdefault((arch, R, gname, s, 's_ctx', '', bn), {})[c] = float(bm[j])
                    # ---- gains
                    items = [(name.format(), h, a.format(h=h), b.format(h=h)) for name, a, b in TRANS for h in range(4)] + \
                            [(name, '', a, b) for name, a, b in TRANS_ALL]
                    for name, h, a, b in items:
                        g_ring = np.mean([np.log10(z[f's{s}b{k}|{b}|ringP'] / z[f's{s}b{k}|{a}|ringP']) for k in bl], 0)
                        g_band = np.mean([np.log10(z[f's{s}b{k}|{b}|bandP'] / z[f's{s}b{k}|{a}|bandP']) for k in bl], 0)
                        head = '' if h == '' else f'k={3 + 2 * h}'
                        for gname, idx in groups.items():
                            m, p25, p75 = stats(g_ring[idx])
                            for r in range(len(rho)):
                                gain_rows.append((arch, c, R, gname, s, name, head, r, f'{rho[r]:.5g}', f'{m[r]:.5g}',
                                                  f'{p25[r]:.5g}', f'{p75[r]:.5g}'))
                            bm, b25, b75 = stats(g_band[idx])
                            for j, bn in enumerate(BANDS):
                                band_rows.append((arch, c, R, gname, s, f'gain {name}', head, bn, f'{bm[j]:.6g}',
                                                  f'{b25[j]:.6g}', f'{b75[j]:.6g}'))
                                vals.setdefault((arch, R, gname, s, f'gain {name}', head, bn), {})[c] = float(bm[j])
                    # ---- context map
                    mp = np.mean([z[f'map|s{s}b{b}'] for b in bl], 0)                     # (N, H, W)
                    up = F.interpolate(torch.from_numpy(mp)[:, None], size=(R, R), mode='bilinear',
                                       align_corners=False)[:, 0].numpy()
                    stage_maps[(c, s)] = up
            np.savez_compressed(os.path.join(HERE, 'data', f'stage_maps_R{R}_{arch}.npz'),
                                **{f'{c}|s{s}': v.astype(np.float32) for (c, s), v in stage_maps.items()})
            # ---- map correlation and object ratio
            stages = sorted({s for (_, s) in stage_maps})
            pairs = [('IN1K', c) for c in COCO + ADE] + [tuple(COCO), tuple(ADE)] + list(itertools.product(COCO, ADE))
            ptype = lambda a, b: 'vs IN1K' if a == 'IN1K' else ('within task' if (a in COCO) == (b in COCO) else 'cross task')
            for s in stages:
                for i, m in enumerate(meta):
                    for a, b in pairs:
                        if (a, s) in stage_maps and (b, s) in stage_maps:
                            r = np.corrcoef(stage_maps[(a, s)][i].ravel(), stage_maps[(b, s)][i].ravel())[0, 1]
                            corr_rows.append((arch, R, s, i, m['group'], m['file'], f'{a}~{b}', ptype(a, b), f'{r:.6g}'))
                    if m['has_mask']:
                        mk = masks[i]
                        for c in CKPTS:
                            if (c, s) not in stage_maps:
                                continue
                            e = stage_maps[(c, s)][i]
                            area = float(mk.mean())
                            ef = float(e[mk].sum() / e.sum()) if area > 0 else float('nan')
                            obj_rows.append((arch, R, s, c, i, m['group'], m['file'], f'{area:.6g}', f'{ef:.6g}',
                                             f'{ef / area:.6g}' if area > 0 else 'nan'))
    # ---- contrast
    summary = {}
    for (arch, R, gname, s, q, head, bn), v in vals.items():
        if not all(c in v for c in CKPTS):
            continue
        W = max(abs(v[COCO[0]] - v[COCO[1]]), abs(v[ADE[0]] - v[ADE[1]]))
        X = [abs(v[a] - v[b]) for a in COCO for b in ADE]
        Xm = abs(np.mean([v[a] for a in COCO]) - np.mean([v[b] for b in ADE]))
        V = [abs(v[c] - v['IN1K']) for c in COCO + ADE]
        contrast_rows.append((arch, R, gname, s, q, head, bn, *(f'{v[c]:.6g}' for c in CKPTS), f'{W:.6g}', f'{min(X):.6g}',
                              f'{max(X):.6g}', f'{Xm:.6g}', f'{min(V):.6g}', f'{max(V):.6g}', int(min(X) > W)))
        fam = q.split()[0] + (' ' + q.split()[1] if q.startswith('gain') else '')
        for key in ((arch, R, gname, fam), (arch, R, gname, 'all quantities')):
            d = summary.setdefault(key, [0, 0, []])
            d[0] += 1
            d[1] += int(min(X) > W)
            if W > 0:
                d[2].append(min(X) / W)

    def wcsv(name, header, rows, gz=False):
        op = gzip.open if gz else open
        with op(os.path.join(HERE, 'data', name), 'wt', newline='') as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
    wcsv('profiles.csv.gz', ['arch', 'ckpt', 'res', 'group', 'stage', 'tensor', 'ring', 'rho', 'mean', 'p25', 'p75'],
         prof_rows, gz=True)
    wcsv('transfer_gain.csv.gz', ['arch', 'ckpt', 'res', 'group', 'stage', 'transition', 'head', 'ring', 'rho', 'mean',
                                  'p25', 'p75'], gain_rows, gz=True)
    wcsv('band_values.csv', ['arch', 'ckpt', 'res', 'group', 'stage', 'quantity', 'head', 'band', 'mean', 'p25', 'p75'],
         band_rows)
    wcsv('contrast.csv', ['arch', 'res', 'group', 'stage', 'quantity', 'head', 'band', *CKPTS, 'W_within_max',
                          'X_cross_min', 'X_cross_max', 'X_groupmean', 'vsIN1K_min', 'vsIN1K_max', 'cross_gt_within'],
         contrast_rows)
    wcsv('contrast_summary.csv', ['arch', 'res', 'group', 'quantity', 'cells', 'cross_min_gt_within_max',
                                  'median_Xmin_over_W'],
         [(*k, v[0], v[1], f'{np.median(v[2]):.4g}' if v[2] else 'nan') for k, v in sorted(summary.items(), key=str)])
    wcsv('map_correlation.csv', ['arch', 'res', 'stage', 'image', 'group', 'file', 'pair', 'pair_type', 'pearson_r'],
         corr_rows)
    agg = {}
    for arch, R, s, i, g, f_, pair, pt, r in corr_rows:
        for gg in (g, 'all'):
            agg.setdefault((arch, R, s, gg, pt), []).append(float(r))
    wcsv('map_correlation_summary.csv', ['arch', 'res', 'stage', 'group', 'pair_type', 'n', 'mean_r', 'p25', 'p75'],
         [(*k, len(v), f'{np.mean(v):.4f}', f'{np.percentile(v, 25):.4f}', f'{np.percentile(v, 75):.4f}')
          for k, v in sorted(agg.items(), key=str)])
    wcsv('object_ratio.csv', ['arch', 'res', 'stage', 'ckpt', 'image', 'group', 'file', 'object_area_fraction',
                              'context_energy_fraction_in_object', 'energy_over_area'], obj_rows)
    print('rows', len(prof_rows), len(gain_rows), len(band_rows), len(contrast_rows), len(corr_rows), len(obj_rows))


if __name__ == '__main__':
    main()
