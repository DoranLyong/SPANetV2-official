#!/usr/bin/env python
"""Run the classification experiments over the GPUs: one job per GPU at a time, jobs in priority order.

    python scheduler.py --gpus 0 1 2 3 --exps clean band fourier pgd [--models ...]

A job (exp, model) is skipped when data/raw/<exp>_<model>.json exists. A job whose weight file is not downloaded yet
is put back at the end of the queue (download_weights.py may still be running).
"""
import argparse
import json
import os
import queue
import subprocess
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = '/home/server/miniforge3/envs/spanetv2/bin/python'
ORDER = ['spanetv2_s18_pure', 'spanetv2_s18_hybrid', 'convformer_s18', 'caformer_s18', 'dfformer_s18',
         'cdfformer_s18', 'convnext_tiny', 'swin_tiny', 'deit_small']


def weight_ready(model):
    p = os.path.join(HERE, 'data', 'weights_manifest.json')
    if not os.path.exists(p):
        return False
    m = json.load(open(p))
    return model in m and os.path.exists(os.path.join(HERE, m[model]['path']))


def log(msg):
    line = f'[{time.strftime("%F %T")}] {msg}'
    print(line, flush=True)


def worker(gpu, q):
    while True:
        try:
            exp, model, tries = q.get_nowait()
        except queue.Empty:
            return
        done = os.path.join(HERE, 'data', 'raw', f'{exp}_{model}.json')
        lock = os.path.join(HERE, 'logs', f'{exp}_{model}.running')
        if os.path.exists(done) or os.path.exists(lock):
            continue
        if not weight_ready(model):
            if tries > 120:
                log(f'GPU{gpu} give up {exp}:{model} (no weight)')
                continue
            q.put((exp, model, tries + 1))
            time.sleep(30)
            continue
        open(lock, 'w').write(str(gpu))
        log(f'GPU{gpu} start {exp}:{model}')
        t0 = time.time()
        with open(os.path.join(HERE, 'logs', f'{exp}_{model}.log'), 'w') as f:
            r = subprocess.run([PY, '-W', 'ignore', 'exp_cls.py', exp, '--model', model], cwd=HERE, stdout=f,
                               stderr=subprocess.STDOUT, env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu)})
        os.remove(lock)
        log(f'GPU{gpu} end   {exp}:{model} exit={r.returncode} {time.time() - t0:.0f}s')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpus', nargs='+', type=int, default=[0, 1, 2, 3])
    ap.add_argument('--exps', nargs='+', default=['clean', 'band', 'fourier', 'pgd'])
    ap.add_argument('--models', nargs='+', default=ORDER)
    args = ap.parse_args()
    q = queue.Queue()
    for e in args.exps:
        for m in args.models + (['convnext_tiny_ema'] if e == 'clean' else []):
            q.put((e, m, 0))
    ts = [threading.Thread(target=worker, args=(g, q)) for g in args.gpus]
    for t in ts:
        t.start()
        time.sleep(2)
    for t in ts:
        t.join()
    log('ALL DONE')


if __name__ == '__main__':
    main()
