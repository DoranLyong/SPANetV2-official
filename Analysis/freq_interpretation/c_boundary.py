#!/usr/bin/env python
"""C. Decomposing the boundary results (cached predictions of ../freq_robustness; CPU).

    python c_boundary.py [--workers 24]

ADE20K (UperNet, Semantic FPN, every model): per image and class, with the ground-truth boundary band
B = union over classes c of mask_to_boundary(gt == c, 0.02) (boundary-iou-api, 2 % of the diagonal) and the
valid pixels V (gt != 255):
  full      I/U of (pred == c, gt == c) on V                       -> mIoU
  boundary  Boundary IoU terms as in freq_robustness boundary_eval -> Boundary IoU
  interior  I/U of (pred == c, gt == c) on V outside B             -> interior IoU (class mean)
Thin-structure subset: freq_robustness boundary_eval.thin_score (fraction of labelled pixels removed by a per-class
5x5 opening) for all 2,000 images; top 10 % (200 images) vs the other 1,800. Metrics of a subset are recomputed from the
per-image sums (class mean over classes with union > 0 in that subset).
COCO: boundary AP and mask AP by object size (APs / APm / APl, boundary-iou-api COCOeval areaRng) from
freq_robustness data/raw/boundary_ap_*.json.
"""
import argparse
import csv
import importlib.util
import json
import os
from multiprocessing import Pool

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FR = os.path.join(os.path.dirname(HERE), 'freq_robustness')
_spec = importlib.util.spec_from_file_location('_be', os.path.join(FR, 'boundary_eval.py'))
BE = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(BE)
NC = 150
SEG = ['upernet_spanetv2_s18_pure', 'upernet_spanetv2_s18_hybrid', 'upernet_spanetv2_s36_pure', 'upernet_spanetv2_s36_hybrid',
       'upernet_convnext_tiny', 'upernet_convnext_small', 'upernet_swin_tiny', 'upernet_swin_small',
       'fpn_spanetv2_s18_pure', 'fpn_spanetv2_s18_hybrid', 'fpn_spanetv2_s36_pure', 'fpn_spanetv2_s36_hybrid',
       'fpn_poolformer_s24', 'fpn_poolformer_s36']
SEG_PAIRS = {  # (setting, size): (SPANetV2 models, comparison models)
    ('upernet', 'S18'): (['upernet_spanetv2_s18_pure', 'upernet_spanetv2_s18_hybrid'], ['upernet_convnext_tiny', 'upernet_swin_tiny']),
    ('upernet', 'S36'): (['upernet_spanetv2_s36_pure', 'upernet_spanetv2_s36_hybrid'], ['upernet_convnext_small', 'upernet_swin_small']),
    ('fpn', 'S18'): (['fpn_spanetv2_s18_pure', 'fpn_spanetv2_s18_hybrid'], ['fpn_poolformer_s24']),
    ('fpn', 'S36'): (['fpn_spanetv2_s36_pure', 'fpn_spanetv2_s36_hybrid'], ['fpn_poolformer_s36']),
}
DET_PAIRS = {
    ('mask_rcnn', 'S18'): (['mask_rcnn_spanetv2_s18_pure', 'mask_rcnn_spanetv2_s18_hybrid'], ['mask_rcnn_convnext_tiny', 'mask_rcnn_swin_tiny']),
    ('cascade', 'S18'): (['cascade_spanetv2_s18_pure', 'cascade_spanetv2_s18_hybrid'], ['cascade_convnext_tiny', 'cascade_swin_tiny']),
    ('cascade', 'S36'): (['cascade_spanetv2_s36_pure', 'cascade_spanetv2_s36_hybrid'], ['cascade_convnext_small', 'cascade_swin_small']),
}


def per_image(stem):
    gt = BE.load_gt(stem)
    valid = gt != 255
    gtb = {c: BE.mask_to_boundary((gt == c).astype(np.uint8), BE.DIL).astype(bool) for c in np.unique(gt[valid])}
    band = np.zeros_like(valid)
    for v in gtb.values():
        band |= v
    inner = valid & ~band
    out = {}
    for m in SEG:
        pred = cv2.imread(os.path.join(FR, 'cache', 'preds', 'seg', m, stem + '.png'), cv2.IMREAD_UNCHANGED).astype(np.int32)
        A = np.zeros((6, NC), np.int64)        # I, U, BI, BU, intI, intU
        p, g = pred[valid], gt[valid]
        ai = np.bincount(p[p == g], minlength=NC)[:NC]
        A[0], A[1] = ai, np.bincount(p, minlength=NC)[:NC] + np.bincount(g, minlength=NC)[:NC] - ai
        pi, gi = pred[inner], gt[inner]
        ii = np.bincount(pi[pi == gi], minlength=NC)[:NC]
        A[4], A[5] = ii, np.bincount(pi, minlength=NC)[:NC] + np.bincount(gi, minlength=NC)[:NC] - ii
        for c in np.union1d(np.unique(g), np.unique(p)):
            if c >= NC:
                continue
            gb = gtb.get(c)
            if gb is None:
                gb = np.zeros_like(valid)
            pb = BE.mask_to_boundary((pred == c).astype(np.uint8), BE.DIL).astype(bool)
            A[2, c] = int((gb & pb & valid).sum())
            A[3, c] = int(((gb | pb) & valid).sum())
        out[m] = A
    return stem, out, BE.thin_score(stem), float(inner.sum() / max(valid.sum(), 1))


def miou(I, U):
    m = U > 0
    return float((I[m] / U[m]).mean() * 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=24)
    args = ap.parse_args()
    stems = BE.stems()
    with Pool(args.workers) as pool:
        res = pool.map(per_image, stems, chunksize=8)
    thin = np.array([r[2] for r in res])
    inner_frac = np.array([r[3] for r in res])
    order = np.argsort(-thin)
    top = np.zeros(len(stems), bool)
    top[order[:len(stems) // 10]] = True
    subsets = {'all': np.ones(len(stems), bool), 'thin_top10pct': top, 'rest': ~top}
    rows, metrics = [], {}
    for m in SEG:
        A = np.stack([r[1][m] for r in res])                                     # (N, 6, NC)
        for sname, sel in subsets.items():
            S = A[sel].sum(0)
            v = dict(mIoU=miou(S[0], S[1]), boundary_IoU=miou(S[2], S[3]), interior_IoU=miou(S[4], S[5]))
            metrics[(m, sname)] = v
            rows.append((m, sname, int(sel.sum()), *(f'{v[k]:.4f}' for k in ('mIoU', 'boundary_IoU', 'interior_IoU'))))
    gains = []
    for (setting, size), (ours, others) in SEG_PAIRS.items():
        for a in ours:
            for b in others:
                for sname in subsets:
                    va, vb = metrics[(a, sname)], metrics[(b, sname)]
                    gains.append((setting, size, a, b, sname, *(f'{va[k] - vb[k]:.4f}' for k in ('mIoU', 'boundary_IoU', 'interior_IoU'))))
    det = []
    for (fw, size), (ours, others) in DET_PAIRS.items():
        for a in ours:
            for b in others:
                ja = json.load(open(os.path.join(FR, 'data', 'raw', f'boundary_ap_{a}.json')))
                jb = json.load(open(os.path.join(FR, 'data', 'raw', f'boundary_ap_{b}.json')))
                for k in ('AP', 'APs', 'APm', 'APl'):
                    det.append((fw, size, a, b, k, f"{ja['boundary'][k]:.4f}", f"{jb['boundary'][k]:.4f}",
                                f"{ja['boundary'][k] - jb['boundary'][k]:.4f}", f"{ja['segm_same_api'][k]:.4f}",
                                f"{jb['segm_same_api'][k]:.4f}", f"{ja['segm_same_api'][k] - jb['segm_same_api'][k]:.4f}"))

    def w(name, header, rr):
        with open(os.path.join(HERE, 'data', name), 'w', newline='') as f:
            cw = csv.writer(f)
            cw.writerow(header)
            cw.writerows(rr)
    w('c_seg_metrics.csv', ['model', 'subset', 'n_images', 'mIoU', 'boundary_IoU', 'interior_IoU'], rows)
    w('c_seg_gains.csv', ['setting', 'size', 'spanetv2', 'comparison', 'subset', 'gain_mIoU', 'gain_boundary_IoU',
                          'gain_interior_IoU'], gains)
    w('c_det_size.csv', ['framework', 'size', 'spanetv2', 'comparison', 'metric', 'boundary_spanetv2', 'boundary_comparison',
                         'gain_boundary', 'mask_spanetv2', 'mask_comparison', 'gain_mask'], det)
    json.dump(dict(thin_threshold_top10=float(thin[order[len(stems) // 10 - 1]]), thin_median=float(np.median(thin)),
                   interior_fraction_mean=float(inner_frac.mean()),
                   thin_top10_images=[stems[i] for i in order[:len(stems) // 10]]),
              open(os.path.join(HERE, 'data', 'c_subsets.json'), 'w'), indent=1)
    print('done', len(rows), len(gains), len(det))


if __name__ == '__main__':
    main()
