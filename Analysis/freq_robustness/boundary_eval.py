#!/usr/bin/env python
"""D1 / D2 metrics from the saved predictions (seg_eval.py): mIoU and Boundary IoU on ADE20K val.

    python boundary_eval.py --models upernet_spanetv2_s18_pure ... [--workers 16]
    python boundary_eval.py --select-examples          # 6 thin-structure images (fixed seed) -> data/boundary_examples.json

Boundary IoU (Cheng et al., CVPR 2021): for masks G (ground truth) and P (prediction),
    BIoU = |(G_d n G) n (P_d n P)| / |(G_d n G) u (P_d n P)|,  G_d n G = mask_to_boundary(G, d)
with mask_to_boundary from the official boundary-iou-api (boundary_iou/utils/boundary_utils.py; 3x3 erosion repeated
d = round(0.02 * image diagonal) times, the paper's default 2 %). The API has no semantic-segmentation mode, so it is
applied per class like mIoU: for each class c, G = (gt == c), P = (pred == c) in every image; intersections and unions
of the boundary regions are summed over the whole validation set (pixels labelled 255 / 'other' in the ground truth are
not counted), BIoU_c = I_c / U_c, and the mean over classes with U_c > 0 is reported.
"""
import argparse
import importlib.util
import json
import os
from multiprocessing import Pool

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ADE = os.path.join(HERE, 'cache', 'datasets', 'ade20k', 'ADEChallengeData2016')
PREDS = os.path.join(HERE, 'cache', 'preds', 'seg')
RAW = os.path.join(HERE, 'data', 'raw')
DIL = 0.02
NC = 150

_spec = importlib.util.spec_from_file_location(
    '_biou_utils', os.path.join(HERE, 'third_party', 'boundary-iou-api', 'boundary_iou', 'utils', 'boundary_utils.py'))
_bu = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bu)
mask_to_boundary = _bu.mask_to_boundary


def load_gt(stem):
    g = cv2.imread(os.path.join(ADE, 'annotations', 'validation', stem + '.png'), cv2.IMREAD_UNCHANGED).astype(np.int32)
    g = g - 1                     # reduce_zero_label: 0 (other) -> 255, 1..150 -> 0..149
    g[g < 0] = 255
    return g


def stems():
    return sorted(os.path.splitext(f)[0] for f in os.listdir(os.path.join(ADE, 'images', 'validation')))


def per_image(args):
    model, stem = args
    gt = load_gt(stem)
    pred = cv2.imread(os.path.join(PREDS, model, stem + '.png'), cv2.IMREAD_UNCHANGED).astype(np.int32)
    assert pred.shape == gt.shape
    valid = gt != 255
    bi = np.zeros(NC, np.int64)
    bu = np.zeros(NC, np.int64)
    p, g = pred[valid], gt[valid]
    ai = np.bincount(p[p == g], minlength=NC)[:NC]
    au = np.bincount(p, minlength=NC)[:NC] + np.bincount(g, minlength=NC)[:NC] - ai
    for c in np.union1d(np.unique(g), np.unique(p)):
        if c >= NC:
            continue
        gb = mask_to_boundary((gt == c).astype(np.uint8), DIL).astype(bool)
        pb = mask_to_boundary((pred == c).astype(np.uint8), DIL).astype(bool)
        bi[c] += int((gb & pb & valid).sum())
        bu[c] += int(((gb | pb) & valid).sum())
    return ai, au, bi, bu


def evaluate(model, workers):
    ss = stems()
    with Pool(workers) as pool:
        res = pool.map(per_image, [(model, s) for s in ss], chunksize=8)
    ai, au, bi, bu = (np.sum([r[k] for r in res], axis=0) for k in range(4))
    iou = np.where(au > 0, ai / np.maximum(au, 1), np.nan)
    biou = np.where(bu > 0, bi / np.maximum(bu, 1), np.nan)
    return dict(model=model, n_images=len(ss), mIoU=float(np.nanmean(iou) * 100),
                boundary_IoU=float(np.nanmean(biou) * 100), dilation_ratio=DIL,
                per_class_IoU=[None if np.isnan(x) else float(x) for x in iou],
                per_class_boundary_IoU=[None if np.isnan(x) else float(x) for x in biou])


def thin_score(stem):
    """Fraction of labelled pixels removed by a per-class 5x5 morphological opening (structures thinner than 5 px)."""
    gt = load_gt(stem)
    valid = gt != 255
    removed = np.zeros_like(valid)
    k = np.ones((5, 5), np.uint8)
    for c in np.unique(gt[valid]):
        m = (gt == c).astype(np.uint8)
        removed |= (m.astype(bool) & ~cv2.morphologyEx(m, cv2.MORPH_OPEN, k).astype(bool))
    return float(removed.sum() / max(valid.sum(), 1))


def select_examples(workers, top=60, k=6, seed=0):
    ss = stems()
    with Pool(workers) as pool:
        sc = pool.map(thin_score, ss, chunksize=16)
    order = np.argsort(sc)[::-1][:top]
    pick = np.sort(np.random.default_rng(seed).choice(order, k, replace=False))
    out = dict(rule=f'thin-structure score = fraction of labelled pixels removed by a per-class 5x5 opening; '
                    f'top {top} of {len(ss)} images, then {k} drawn with numpy default_rng({seed})',
               images=[ss[i] for i in pick], scores=[sc[i] for i in pick],
               top_threshold=float(sc[order[-1]]), median_score_all=float(np.median(sc)))
    with open(os.path.join(HERE, 'data', 'boundary_examples.json'), 'w') as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='*', default=[])
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--select-examples', action='store_true')
    args = ap.parse_args()
    if args.select_examples:
        select_examples(args.workers)
    for m in args.models:
        r = evaluate(m, args.workers)
        with open(os.path.join(RAW, f'boundary_{m}.json'), 'w') as f:
            json.dump(r, f, indent=1)
        print(m, f"mIoU {r['mIoU']:.2f}  BIoU {r['boundary_IoU']:.2f}", flush=True)


if __name__ == '__main__':
    main()
