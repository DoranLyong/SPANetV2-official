#!/usr/bin/env python
"""Images for the branch-spectra / context study (fixed seeds) at R1 = 224 and R2 = 512.

    python select_images.py [--misc DIR] [--imnet-val DIR] [--coco DIR] [--ade DIR]

Groups
  imagenet : the images of image_classification/misc/imgs. All 10 are ImageNet-1K val files (byte-identical, matched by
             md5 against /home/server/dataset/ImageNet2012/val, see data/images.json). The task asks for 8 per group
             and for the misc images to be included; all 10 are kept so none of them is dropped. Order: numpy
             default_rng(0).permutation(10).
  coco     : 8 COCO val2017 images, default_rng(1).choice over the image ids sorted ascending.
  ade20k   : 8 ADE20K val images, default_rng(2).choice over the sorted file names.
Main-text figures use the first 3 images of each group in this order; the rest go to the appendix.
Preprocessing (both resolutions, every checkpoint): Resize(shorter side = R, PIL bicubic) -> CenterCrop(R); the model
input is (x / 255 - ImageNet mean) / std. Object masks (same geometry, nearest): COCO = union of all instance masks
(crowd included); ADE20K = pixels of the 100 'thing' classes of the Mask2Former ADE20K-150 panoptic split
(data/ade20k_150_isthing.json). ImageNet images have no mask.
Writes data/images.json, data/images_R{224,512}.npy (N, R, R, 3) uint8, data/masks_R{224,512}.npy (N, R, R) bool.
"""
import argparse
import hashlib
import json
import os

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
FRC = os.path.join(os.path.dirname(HERE), 'freq_robustness', 'cache', 'datasets')
RES = (224, 512)


def prep(img, R, resample):
    w, h = img.size
    s = R / min(w, h)
    nw, nh = max(R, round(w * s)), max(R, round(h * s))
    img = img.resize((nw, nh), resample)
    l, t = (nw - R) // 2, (nh - R) // 2
    return img.crop((l, t, l + R, t + R))


def md5(p):
    return hashlib.md5(open(p, 'rb').read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--misc', default=os.path.join(REPO, 'image_classification', 'misc', 'imgs'))
    ap.add_argument('--imnet-val', default='/home/server/dataset/ImageNet2012/val')
    ap.add_argument('--coco', default=os.path.join(FRC, 'coco'))
    ap.add_argument('--ade', default=os.path.join(FRC, 'ade20k', 'ADEChallengeData2016'))
    args = ap.parse_args()
    from pycocotools.coco import COCO
    things = {c['id'] for c in json.load(open(os.path.join(HERE, 'data', 'ade20k_150_isthing.json')))['categories']
              if c['isthing']}

    items = []                                                  # (group, source path, label/meta, mask fn)
    misc = sorted(os.listdir(args.misc), key=lambda f: int(f.split('.')[0]))
    by_size = {}
    for w in os.listdir(args.imnet_val):
        for f in os.listdir(os.path.join(args.imnet_val, w)):
            p = os.path.join(args.imnet_val, w, f)
            by_size.setdefault(os.path.getsize(p), []).append(p)
    for k in np.random.default_rng(0).permutation(len(misc)):
        p = os.path.join(args.misc, misc[k])
        match = [os.path.relpath(c, args.imnet_val) for c in by_size.get(os.path.getsize(p), []) if md5(c) == md5(p)]
        items.append(dict(group='imagenet', path=p, file=f'misc/imgs/{misc[k]}', imagenet_val_file=match, mask=None))
    coco = COCO(os.path.join(args.coco, 'annotations', 'instances_val2017.json'))
    ids = sorted(coco.getImgIds())
    for i in np.random.default_rng(1).choice(len(ids), 8, replace=False):
        info = coco.loadImgs(ids[i])[0]
        anns = coco.loadAnns(coco.getAnnIds(imgIds=ids[i], iscrowd=None))
        m = np.zeros((info['height'], info['width']), bool)
        for a in anns:
            m |= coco.annToMask(a).astype(bool)
        items.append(dict(group='coco', path=os.path.join(args.coco, 'val2017', info['file_name']),
                          file=info['file_name'], coco_image_id=ids[i], n_instances=len(anns), mask=m))
    ade_files = sorted(f for f in os.listdir(os.path.join(args.ade, 'images', 'validation')) if f.endswith('.jpg'))
    for i in np.random.default_rng(2).choice(len(ade_files), 8, replace=False):
        f = ade_files[i]
        g = np.asarray(Image.open(os.path.join(args.ade, 'annotations', 'validation', f.replace('.jpg', '.png')))).astype(int) - 1
        m = np.isin(g, list(things))
        items.append(dict(group='ade20k', path=os.path.join(args.ade, 'images', 'validation', f), file=f, mask=m))

    rank = {}
    meta = []
    out = {R: ([], []) for R in RES}
    for it in items:
        rank[it['group']] = rank.get(it['group'], -1) + 1
        img = Image.open(it['path']).convert('RGB')
        for R in RES:
            out[R][0].append(np.asarray(prep(img, R, Image.BICUBIC)))
            if it['mask'] is None:
                out[R][1].append(np.zeros((R, R), bool))
            else:
                mi = Image.fromarray(it['mask'].astype(np.uint8) * 255)
                out[R][1].append(np.asarray(prep(mi, R, Image.NEAREST)) > 127)
        rec = {k: v for k, v in it.items() if k not in ('mask', 'path')}
        rec.update(original_size=list(img.size), has_mask=it['mask'] is not None,
                   figure='main' if rank[it['group']] < 3 else 'appendix',
                   object_area_R224=float(out[224][1][-1].mean()) if it['mask'] is not None else None)
        meta.append(rec)
    for R in RES:
        np.save(os.path.join(HERE, 'data', f'images_R{R}.npy'), np.stack(out[R][0]))
        np.save(os.path.join(HERE, 'data', f'masks_R{R}.npy'), np.stack(out[R][1]))
    json.dump(dict(rule=__doc__.split('Writes')[0].strip(), images=meta),
              open(os.path.join(HERE, 'data', 'images.json'), 'w'), indent=1, default=str)
    for m in meta:
        print(m['group'], m['file'], m['figure'], m.get('imagenet_val_file', ''), m.get('object_area_R224'))


if __name__ == '__main__':
    main()
