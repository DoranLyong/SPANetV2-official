#!/usr/bin/env python
"""D3: COCO Boundary AP (and mask AP with the same API) from det_eval.py results.

    python boundary_ap.py --models cascade_spanetv2_s18_pure ...

Uses the official boundary-iou-api (third_party/boundary-iou-api, imported from the clone, not installed), exactly as its
tools/coco_instance_evaluation.py: COCO(gt, get_boundary=True, dilation_ratio=0.02), loadRes(results.segm.json),
COCOeval(iouType='boundary') -> evaluate / accumulate / summarize. Boundary AP uses min(mask IoU, boundary IoU)
for every ground-truth / detection pair (Cheng et al., CVPR 2021). The 'segm' run of the same API is recorded as a check
of the mask AP computed by mmdet's CocoMetric. The API code is not changed; only `np.float = float` is set in this process
because the API uses the alias that numpy >= 1.24 removed.
"""
import argparse
import contextlib
import io
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'third_party', 'boundary-iou-api'))
if not hasattr(np, 'float'):        # the API (2021) uses np.float, removed in numpy 1.24; alias restored in-process only
    np.float = float
from boundary_iou.coco_instance_api.coco import COCO  # noqa: E402
from boundary_iou.coco_instance_api.cocoeval import COCOeval  # noqa: E402

GT = os.path.join(HERE, 'cache', 'datasets', 'coco', 'annotations', 'instances_val2017.json')
NAMES = ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl']


def run(gt, dt_file, iou_type):
    dt = gt.loadRes(dt_file)
    ev = COCOeval(gt, dt, iouType=iou_type, dilation_ratio=0.02)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return {n: float(v) * 100 for n, v in zip(NAMES, ev.stats[:6])}, buf.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', required=True)
    args = ap.parse_args()
    gt_b = COCO(GT, get_boundary=True, dilation_ratio=0.02)
    for m in args.models:
        dt = os.path.join(HERE, 'cache', 'preds', 'det', m, 'results.segm.json')
        b, log_b = run(gt_b, dt, 'boundary')
        s, log_s = run(gt_b, dt, 'segm')
        out = dict(model=m, dilation_ratio=0.02, boundary=b, segm_same_api=s)
        with open(os.path.join(HERE, 'data', 'raw', f'boundary_ap_{m}.json'), 'w') as f:
            json.dump(out, f, indent=1)
        with open(os.path.join(HERE, 'logs', f'boundary_ap_{m}.log'), 'w') as f:
            f.write(log_b + '\n' + log_s)
        print(m, 'boundary AP', round(b['AP'], 2), 'mask AP', round(s['AP'], 2), flush=True)


if __name__ == '__main__':
    main()
