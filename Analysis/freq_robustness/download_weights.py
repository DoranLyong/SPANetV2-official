#!/usr/bin/env python
"""Download the public weights used here; every file goes into its own new directory weights/<group>/<stem>/<file>.

    python download_weights.py cls            # classification (ImageNet-1K) weights
    python download_weights.py seg det        # downstream weights (section D)
    python download_weights.py all

A file that is already present with the expected size is kept. SPANetV2 release files that were already downloaded
for ../srf_task_analysis (same GitHub release asset, same size) are hard-linked instead of downloaded again.
Every file is fetched as byte ranges over at most --conn connections and resumed after a reset.
Writes data/weights_manifest.json (url, path, size, sha256).
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
SRF = os.path.join(os.path.dirname(HERE), 'srf_task_analysis', 'ckpts')
REL = 'https://github.com/DoranLyong/SPANetV2-official/releases/download'
UA = {'User-Agent': 'freq-robustness-downloader'}

WEIGHTS = {
    'cls': {
        'spanetv2_s18_pure': f'{REL}/ImageNet-1K/spanetv2_s18_pure_res-scale_Full-ExSPAM.pth',
        'spanetv2_s18_hybrid': f'{REL}/ImageNet-1K/spanetv2_s18_hybrid_k7_res-scale_Full-ExSPAM.pth',
        'convformer_s18': 'https://huggingface.co/sail/dl/resolve/main/convformer/convformer_s18.pth',
        'caformer_s18': 'https://huggingface.co/sail/dl/resolve/main/caformer/caformer_s18.pth',
        'dfformer_s18': 'https://github.com/okojoalg/dfformer/releases/download/weights/dfformer_s18.pth',
        'cdfformer_s18': 'https://github.com/okojoalg/dfformer/releases/download/weights/cdfformer_s18.pth',
        'convnext_tiny': 'https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224.pth',
        'convnext_tiny_ema': 'https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth',
        'swin_tiny': 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth',
        'deit_small': 'https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth',
    },
    'seg': {
        **{f'upernet_spanetv2_{m}': f'{REL}/ADE20K/upernet_spanetv2_{m}_ade20k_{t}.pth'
           for m, t in [('s18_pure', '4872e-2'), ('s18_hybrid', '4913e-2'), ('s36_pure', '4976e-2'),
                        ('s36_hybrid', '5163e-2')]},
        **{f'fpn_spanetv2_{m}': f'{REL}/ADE20K/fpn_spanetv2_{m}_ade20k_{t}.pth'
           for m, t in [('s18_pure', '467e-1'), ('s18_hybrid', '478e-1'), ('s36_pure', '479e-1'),
                        ('s36_hybrid', '486e-1')]},
        'upernet_convnext_tiny': 'https://dl.fbaipublicfiles.com/convnext/ade20k/upernet_convnext_tiny_1k_512x512.pth',
        'upernet_convnext_small': 'https://dl.fbaipublicfiles.com/convnext/ade20k/upernet_convnext_small_1k_512x512.pth',
        'upernet_swin_tiny': 'https://github.com/SwinTransformer/storage/releases/download/v1.0.1/'
                             'upernet_swin_tiny_patch4_window7_512x512.pth',
        'upernet_swin_small': 'https://github.com/SwinTransformer/storage/releases/download/v1.0.1/'
                              'upernet_swin_small_patch4_window7_512x512.pth',
        # sail-sg/poolformer links its FPN weights on Google Drive only, which this server cannot reach
        # (SSL_ERROR_SYSCALL); the mmseg model zoo files are used instead (report.md D2).
        'fpn_poolformer_s24': 'https://download.openmmlab.com/mmsegmentation/v0.5/poolformer/'
                              'fpn_poolformer_s24_8x4_512x512_40k_ade20k/'
                              'fpn_poolformer_s24_8x4_512x512_40k_ade20k_20220503_222049-394a7cf7.pth',
        'fpn_poolformer_s36': 'https://download.openmmlab.com/mmsegmentation/v0.5/poolformer/'
                              'fpn_poolformer_s36_8x4_512x512_40k_ade20k/'
                              'fpn_poolformer_s36_8x4_512x512_40k_ade20k_20220501_151122-b47e607d.pth',
    },
    'det': {
        'mask_rcnn_spanetv2_s18_pure': f'{REL}/COCO2017/mask-rcnn_spanetv2_s18_pure_3x_epoch_35.pth',
        'mask_rcnn_spanetv2_s18_hybrid': f'{REL}/COCO2017/mask-rcnn_spanetv2_s18_hybrid_3x_epoch_36.pth',
        'cascade_spanetv2_s18_pure': f'{REL}/COCO2017/cascade-mask-rcnn_spanetv2_s18_pure_3x_epoch_35.pth',
        'cascade_spanetv2_s18_hybrid': f'{REL}/COCO2017/cascade-mask-rcnn_spanetv2_s18_hybrid_3x_epoch_35.pth',
        'cascade_spanetv2_s36_pure': f'{REL}/COCO2017/cascade-mask-rcnn_spanetv2_s36_pure_3x_epoch_36.pth',
        'cascade_spanetv2_s36_hybrid': f'{REL}/COCO2017/cascade-mask-rcnn_spanetv2_s36_hybrid_3x_epoch_35.pth',
        'mask_rcnn_convnext_tiny': 'https://dl.fbaipublicfiles.com/convnext/coco/mask_rcnn_convnext_tiny_1k_3x.pth',
        'cascade_convnext_tiny': 'https://dl.fbaipublicfiles.com/convnext/coco/cascade_mask_rcnn_convnext_tiny_1k_3x.pth',
        'cascade_convnext_small': 'https://dl.fbaipublicfiles.com/convnext/coco/cascade_mask_rcnn_convnext_small_1k_3x.pth',
        'mask_rcnn_swin_tiny': 'https://github.com/SwinTransformer/storage/releases/download/v1.0.2/'
                               'mask_rcnn_swin_tiny_patch4_window7.pth',
        'cascade_swin_tiny': 'https://github.com/SwinTransformer/storage/releases/download/v1.0.2/'
                             'cascade_mask_rcnn_swin_tiny_patch4_window7.pth',
        'cascade_swin_small': 'https://github.com/SwinTransformer/storage/releases/download/v1.0.2/'
                              'cascade_mask_rcnn_swin_small_patch4_window7.pth',
    },
}


def path_for(group, url):
    name = url.rsplit('/', 1)[1]
    return os.path.join(HERE, 'weights', group, name.rsplit('.', 1)[0], name)


def head_size(url):
    req = urllib.request.Request(url, method='HEAD', headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers['Content-Length'])


def fetch_range(url, path, a, b, tries=200):
    want = b - a + 1
    for t in range(tries):
        got = os.path.getsize(path) if os.path.exists(path) else 0
        if got == want:
            return True
        if got > want:
            os.remove(path)
            continue
        req = urllib.request.Request(url, headers={**UA, 'Range': f'bytes={a + got}-{b}'})
        try:
            with urllib.request.urlopen(req, timeout=60) as r, open(path, 'ab') as w:
                if r.status != 206:
                    raise IOError(f'HTTP {r.status}')
                for buf in iter(lambda: r.read(1 << 20), b''):
                    w.write(buf)
        except Exception as e:
            time.sleep(min(30, 2 + t))
            if t % 10 == 9:
                print(f'retry {t + 1} {os.path.basename(path)}: {type(e).__name__}: {e}', flush=True)
    return os.path.exists(path) and os.path.getsize(path) == want


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('groups', nargs='+', choices=['cls', 'seg', 'det', 'all'])
    ap.add_argument('--conn', type=int, default=12)
    ap.add_argument('--chunks', type=int, default=4)
    args = ap.parse_args()
    groups = ['cls', 'seg', 'det'] if 'all' in args.groups else args.groups
    man_path = os.path.join(HERE, 'data', 'weights_manifest.json')
    manifest = json.load(open(man_path)) if os.path.exists(man_path) else {}
    jobs = []
    for g in groups:
        for key, url in WEIGHTS[g].items():
            out = path_for(g, url)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            size = head_size(url)
            entry = dict(group=g, url=url, path=os.path.relpath(out, HERE), size=size)
            if not (os.path.exists(out) and os.path.getsize(out) == size) and url.startswith(REL):
                tag, name = url[len(REL) + 1:].split('/')
                old = os.path.join(SRF, tag, name.rsplit('.', 1)[0], name)
                if os.path.exists(old) and os.path.getsize(old) == size:
                    os.link(old, out)
                    entry['source'] = f'hard link of ../srf_task_analysis/ckpts/{tag}/{name.rsplit(".", 1)[0]}/{name}'
            manifest[key] = entry
            if not (os.path.exists(out) and os.path.getsize(out) == size):
                n = max(1, min(args.chunks, size // (8 << 20)))
                step = -(-size // n)
                jobs.append((key, url, out, size, [(i, i * step, min(size, (i + 1) * step) - 1) for i in range(n)]))
    with cf.ThreadPoolExecutor(args.conn) as ex:
        futs = {key: [ex.submit(fetch_range, url, f'{out}.c{i}', a, b) for i, a, b in rr]
                for key, url, out, size, rr in jobs}
        for key, url, out, size, rr in jobs:
            ok = all(f.result() for f in futs[key])
            if ok:
                with open(out + '.part', 'wb') as w:
                    for i, _, _ in rr:
                        with open(f'{out}.c{i}', 'rb') as r:
                            for buf in iter(lambda: r.read(1 << 24), b''):
                                w.write(buf)
                        os.remove(f'{out}.c{i}')
                os.replace(out + '.part', out)
            print('done' if ok else 'FAIL', key, flush=True)
    for key, e in manifest.items():
        p = os.path.join(HERE, e['path'])
        if os.path.exists(p) and os.path.getsize(p) == e['size'] and 'sha256' not in e:
            e['sha256'] = sha256(p)
    os.makedirs(os.path.dirname(man_path), exist_ok=True)
    with open(man_path, 'w') as f:
        json.dump(manifest, f, indent=1)
    missing = [k for k, e in manifest.items() if 'sha256' not in e]
    print('missing:', missing if missing else 'none')
    sys.exit(1 if missing else 0)


if __name__ == '__main__':
    main()
