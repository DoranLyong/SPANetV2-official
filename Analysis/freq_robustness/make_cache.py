#!/usr/bin/env python
"""Decode the ImageNet-1K validation set once per evaluation preprocessing and store the 224x224 uint8 images.

    python make_cache.py --val /home/server/dataset/ImageNet2012/val [--workers 32]

Every classification model in this study evaluates with
    Resize(shorter side = floor(224 / crop_pct), bicubic)  ->  CenterCrop(224)  ->  ToTensor  ->  Normalize
(timm `create_transform(is_training=False)`, ConvNeXt `build_transform`, Swin `build_transform`, DeiT `build_transform`;
see report.md 0). Only crop_pct differs (1.0 or 0.875), so two caches are written:
    cache/val_crop1.000.npy   (50000, 224, 224, 3) uint8
    cache/val_crop0.875.npy
    cache/val_labels.npy      (50000,) int64, ImageFolder order (sorted wnid -> 0..999, sorted file names)
    cache/val_files.txt
    cache/subset5k_seed0.npy  indices of the 5-per-class sample (numpy default_rng(0)), sorted
ToTensor of these uint8 arrays (x / 255) is exactly the [0,1] tensor the models' own pipelines produce.
"""
import argparse
import math
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image
from torchvision import transforms

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, 'cache')
CROPS = (1.0, 0.875)


def tf(crop_pct):
    return transforms.Compose([
        transforms.Resize(math.floor(224 / crop_pct), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224)])


TFS = {c: tf(c) for c in CROPS}


def work(args):
    i0, paths = args
    out = {c: np.empty((len(paths), 224, 224, 3), np.uint8) for c in CROPS}
    for k, p in enumerate(paths):
        with open(p, 'rb') as f:
            img = Image.open(f).convert('RGB')
        for c in CROPS:
            out[c][k] = np.asarray(TFS[c](img))
    return i0, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--val', required=True)
    ap.add_argument('--workers', type=int, default=32)
    args = ap.parse_args()
    os.makedirs(CACHE, exist_ok=True)
    wnids = sorted(d for d in os.listdir(args.val) if os.path.isdir(os.path.join(args.val, d)))
    assert len(wnids) == 1000
    files, labels = [], []
    for ci, w in enumerate(wnids):
        for fn in sorted(os.listdir(os.path.join(args.val, w))):
            files.append(os.path.join(args.val, w, fn))
            labels.append(ci)
    n = len(files)
    assert n == 50000, n
    np.save(os.path.join(CACHE, 'val_labels.npy'), np.array(labels, np.int64))
    with open(os.path.join(CACHE, 'val_files.txt'), 'w') as f:
        f.write('\n'.join(os.path.relpath(p, args.val) for p in files) + '\n')
    rng = np.random.default_rng(0)
    lab = np.array(labels)
    sub = np.sort(np.concatenate([rng.choice(np.where(lab == c)[0], 5, replace=False) for c in range(1000)]))
    np.save(os.path.join(CACHE, 'subset5k_seed0.npy'), sub)
    mm = {c: np.lib.format.open_memmap(os.path.join(CACHE, f'val_crop{c:.3f}.npy'), 'w+', np.uint8, (n, 224, 224, 3))
          for c in CROPS}
    chunk = 250
    jobs = [(i, files[i:i + chunk]) for i in range(0, n, chunk)]
    done = 0
    with ProcessPoolExecutor(args.workers) as ex:
        for i0, out in ex.map(work, jobs):
            for c in CROPS:
                mm[c][i0:i0 + len(out[c])] = out[c]
            done += len(out[CROPS[0]])
            if done % 5000 == 0:
                print(f'{done}/{n}', flush=True)
    for m in mm.values():
        m.flush()
    print('done')


if __name__ == '__main__':
    main()
