#!/usr/bin/env python
"""Download the release checkpoints used by analyze_srf.py; one new directory per file.

    python download_ckpts.py [--conn 12] [--chunks 8]   -> ckpts/<tag>/<file stem>/<file>

A single connection to the release CDN runs at about 1 MB/s, and many connections get reset, so every file is
fetched as `--chunks` byte ranges with at most `--conn` connections open in total. Each range is appended to
<file>.c<i> and resumed from where it stopped; a leftover <file>.part (a complete prefix) is kept and only the rest
is fetched. Sizes come from the GitHub release API; when the API publishes a sha256 digest it is checked.
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import sys
import time
import urllib.request

REPO = 'DoranLyong/SPANetV2-official'
BASE = f'https://github.com/{REPO}/releases/download'
FILES = [
    # priority 1: S18-pure
    'ImageNet-1K/spanetv2_s18_pure_res-scale_Full-ExSPAM.pth',
    'COCO2017/mask-rcnn_spanetv2_s18_pure_3x_epoch_35.pth',
    'COCO2017/cascade-mask-rcnn_spanetv2_s18_pure_3x_epoch_35.pth',
    'ADE20K/fpn_spanetv2_s18_pure_ade20k_467e-1.pth',
    'ADE20K/upernet_spanetv2_s18_pure_ade20k_4872e-2.pth',
    # priority 2: S36-pure
    'ImageNet-1K/spanetv2_s36_pure_res-scale_Full-ExSPAM.pth',
    'COCO2017/cascade-mask-rcnn_spanetv2_s36_pure_3x_epoch_36.pth',
    'ADE20K/fpn_spanetv2_s36_pure_ade20k_479e-1.pth',
    'ADE20K/upernet_spanetv2_s36_pure_ade20k_4976e-2.pth',
    # priority 3: S18 / S36-hybrid
    'ImageNet-1K/spanetv2_s18_hybrid_k7_res-scale_Full-ExSPAM.pth',
    'COCO2017/mask-rcnn_spanetv2_s18_hybrid_3x_epoch_36.pth',
    'COCO2017/cascade-mask-rcnn_spanetv2_s18_hybrid_3x_epoch_35.pth',
    'ADE20K/fpn_spanetv2_s18_hybrid_ade20k_478e-1.pth',
    'ADE20K/upernet_spanetv2_s18_hybrid_ade20k_4913e-2.pth',
    'ImageNet-1K/spanetv2_s36_hybrid_k7_res-scale_Full-ExSPAM.pth',
    'COCO2017/cascade-mask-rcnn_spanetv2_s36_hybrid_3x_epoch_35.pth',
    'ADE20K/fpn_spanetv2_s36_hybrid_ade20k_486e-1.pth',
    'ADE20K/upernet_spanetv2_s36_hybrid_ade20k_5163e-2.pth',
]
HERE = os.path.dirname(os.path.abspath(__file__))
UA = {'User-Agent': 'srf-task-analysis-downloader'}


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def release_assets():
    out = {}
    for tag in sorted({f.split('/')[0] for f in FILES}):
        req = urllib.request.Request(f'https://api.github.com/repos/{REPO}/releases/tags/{tag}', headers=UA)
        with urllib.request.urlopen(req, timeout=60) as r:
            for a in json.load(r)['assets']:
                out[f'{tag}/{a["name"]}'] = dict(size=a['size'], digest=a.get('digest'))
    return out


def out_path(f):
    tag, name = f.split('/', 1)
    return os.path.join(HERE, 'ckpts', tag, name[:-4], name)


def plan(f, total, n):
    """Byte ranges still missing after <file>.part, split into n chunks (same split every run)."""
    out = out_path(f)
    have = os.path.getsize(out + '.part') if os.path.exists(out + '.part') else 0
    step = -(-(total - have) // n)
    ranges = []
    for i in range(n):
        a = have + i * step
        b = min(a + step - 1, total - 1)
        if a > b:
            break
        ranges.append((i, a, b))
    return ranges


def fetch(f, i, a, b, tries=200):
    path = f'{out_path(f)}.c{i}'
    want = b - a + 1
    for t in range(tries):
        got = os.path.getsize(path) if os.path.exists(path) else 0
        if got == want:
            return True
        if got > want:
            os.remove(path)
            continue
        req = urllib.request.Request(f'{BASE}/{f}', headers={**UA, 'Range': f'bytes={a + got}-{b}'})
        try:
            with urllib.request.urlopen(req, timeout=60) as r, open(path, 'ab') as w:
                if r.status != 206:
                    raise IOError(f'HTTP {r.status} for a range request')
                while True:
                    buf = r.read(1 << 20)
                    if not buf:
                        break
                    w.write(buf)
        except Exception as e:  # reset / timeout: resume from the current size
            time.sleep(min(30, 2 + t))
            if t % 10 == 9:
                log(f'retry {t + 1} {f} c{i}: {type(e).__name__}: {e}')
    return os.path.exists(path) and os.path.getsize(path) == want


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for b in iter(lambda: fh.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--conn', type=int, default=12)
    ap.add_argument('--chunks', type=int, default=8)
    args = ap.parse_args()
    assets = release_assets()
    with open(os.path.join(HERE, 'data', 'release_assets.json'), 'w') as fh:
        json.dump(assets, fh, indent=1)
    todo = {}
    for f in FILES:
        out = out_path(f)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        total = assets[f]['size']
        if os.path.exists(out) and os.path.getsize(out) == total:
            log('have', f)
            continue
        todo[f] = plan(f, total, args.chunks)
        log('get ', f, f'{sum(b - a + 1 for _, a, b in todo[f]):,} bytes left in {len(todo[f])} ranges')
    ok_all = True
    with cf.ThreadPoolExecutor(args.conn) as ex:
        futs = {f: [ex.submit(fetch, f, i, a, b) for i, a, b in r] for f, r in todo.items()}
        for f, fl in futs.items():
            ok = all(x.result() for x in fl)
            out = out_path(f)
            if ok:
                with open(out + '.part', 'ab') as w:
                    for i, _, _ in todo[f]:
                        with open(f'{out}.c{i}', 'rb') as r:
                            while True:
                                buf = r.read(1 << 24)
                                if not buf:
                                    break
                                w.write(buf)
                for i, _, _ in todo[f]:
                    os.remove(f'{out}.c{i}')
                ok = os.path.getsize(out + '.part') == assets[f]['size']
                if ok:
                    os.replace(out + '.part', out)
            dg = assets[f]['digest']
            if ok and dg:
                ok = dg == 'sha256:' + sha256(out)
                log('sha256', 'ok' if ok else 'MISMATCH', f)
            log('done' if ok else 'FAIL', f)
            ok_all &= ok
    log('DONE' if ok_all else 'DONE with failures (rerun to resume)')
    sys.exit(0 if ok_all else 1)


if __name__ == '__main__':
    main()
