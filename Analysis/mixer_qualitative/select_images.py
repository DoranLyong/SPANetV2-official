#!/usr/bin/env python
"""Pick the 12 images (fixed seeds) and store them after the models' evaluation preprocessing.

    python select_images.py [--imnet-cache ../freq_robustness/cache] [--ade ../freq_robustness/cache/datasets/ade20k/ADEChallengeData2016]

All four models (SPANetV2-S18-pure, ConvFormer-S18, DFFormer-S18, CAFormer-S18) evaluate with
Resize(224, bicubic) on the shorter side -> CenterCrop(224) (crop_pct 1.0, see ../freq_robustness/report.md 0), so one
224x224 crop per image serves every model.
  group 'imnet_random' : 4 ImageNet-1K val images, numpy default_rng(0).choice(50000, 4)
  group 'imnet_sobel'  : 4 images drawn with default_rng(1) from the ImageNet val images whose Sobel gradient energy
                         (mean of gx^2 + gy^2 on the grey 224x224 crop, cv2.Sobel ksize 3) is in the top 25 %,
                         excluding the 'imnet_random' images
  group 'ade_random'   : 4 ADE20K val images, default_rng(2).choice(2000, 4) over the sorted file list
Main-text figures use the first 3 images of 'imnet_random' and of 'imnet_sobel' (in drawing order); the other 6
(1 + 1 + 4 ADE20K) go to the appendix figures.
Writes data/images.json and data/images_u8.npy (12, 224, 224, 3).
"""
import argparse
import json
import os

import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

HERE = os.path.dirname(os.path.abspath(__file__))
FR = os.path.join(os.path.dirname(HERE), 'freq_robustness')


def sobel_energy(u8):
    g = cv2.cvtColor(u8, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return float((gx ** 2 + gy ** 2).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--imnet-cache', default=os.path.join(FR, 'cache'))
    ap.add_argument('--ade', default=os.path.join(FR, 'cache', 'datasets', 'ade20k', 'ADEChallengeData2016'))
    args = ap.parse_args()
    X = np.load(os.path.join(args.imnet_cache, 'val_crop1.000.npy'), mmap_mode='r')
    files = open(os.path.join(args.imnet_cache, 'val_files.txt')).read().split()
    labels = np.load(os.path.join(args.imnet_cache, 'val_labels.npy'))
    n = len(files)
    rand = [int(i) for i in np.random.default_rng(0).choice(n, 4, replace=False)]
    energy = np.array([sobel_energy(np.asarray(X[i])) for i in range(n)])
    thr = float(np.percentile(energy, 75))
    pool = np.setdiff1d(np.where(energy >= thr)[0], rand)
    sob = [int(i) for i in np.random.default_rng(1).choice(pool, 4, replace=False)]
    ade_files = sorted(f for f in os.listdir(os.path.join(args.ade, 'images', 'validation')) if f.endswith('.jpg'))
    ade = [int(i) for i in np.random.default_rng(2).choice(len(ade_files), 4, replace=False)]
    tf = transforms.Compose([transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                             transforms.CenterCrop(224)])
    imgs, meta = [], []
    for g, idx in (('imnet_random', rand), ('imnet_sobel', sob)):
        for k, i in enumerate(idx):
            imgs.append(np.asarray(X[i]))
            meta.append(dict(group=g, source='ImageNet-1K val', file=files[i], val_index=i, label=int(labels[i]),
                             sobel_energy=float(energy[i]), figure='main' if k < 3 else 'appendix'))
    for i in ade:
        with open(os.path.join(args.ade, 'images', 'validation', ade_files[i]), 'rb') as f:
            im = Image.open(f).convert('RGB')
        u8 = np.asarray(tf(im))
        imgs.append(u8)
        meta.append(dict(group='ade_random', source='ADE20K val', file=ade_files[i], index=i,
                         sobel_energy=sobel_energy(u8), figure='appendix'))
    np.save(os.path.join(HERE, 'data', 'images_u8.npy'), np.stack(imgs))
    out = dict(rule=__doc__.split('Writes')[0].strip(), sobel_top25_threshold=thr,
               sobel_energy_median_all=float(np.median(energy)), images=meta)
    with open(os.path.join(HERE, 'data', 'images.json'), 'w') as f:
        json.dump(out, f, indent=1)
    for m in meta:
        print(m['group'], m['file'], m['figure'], round(m['sobel_energy'], 4))


if __name__ == '__main__':
    main()
