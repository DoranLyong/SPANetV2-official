#!/usr/bin/env python
"""Run GPU jobs from a job list, one per GPU at a time (lock files logs/<name>.running; a job is skipped when its
output file exists).

    python run_jobs.py --gpus 0 1 2 3 --list a        # A jobs
    python run_jobs.py --gpus 0 1 --list b            # B jobs
"""
import argparse
import os
import queue
import subprocess
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = '/home/server/miniforge3/envs/spanetv2/bin/python'
A_MODELS = ['spanetv2_s18_hybrid', 'caformer_s18', 'spanetv2_s18_pure', 'convformer_s18', 'deit_small']
B_MODELS = ['spanetv2_s18_pure', 'spanetv2_s18_hybrid', 'convformer_s18', 'caformer_s18', 'dfformer_s18', 'cdfformer_s18']


def jobs(name):
    if name == 'a':
        out = []
        for v in ('high', 'full'):
            out += [(f'a_main_{m}_{v}', ['a_pgd.py', 'main', '--model', m, '--variant', v], f'data/a_main_{m}_{v}.json')
                    for m in A_MODELS]
        out += [(f'a_large_{m}', ['a_pgd.py', 'large', '--model', m], f'data/a_large_{m}.json') for m in A_MODELS]
        out += [(f'a_apgd_{m}', ['a_pgd.py', 'apgd', '--model', m], f'data/a_apgd_{m}.json') for m in A_MODELS]
        return out
    if name == 'a2':
        return [(f'a_apgd2_{m}_{v}', ['a_pgd.py', 'apgd2', '--model', m, '--variant', v], f'data/a_apgd2_{m}_{v}.json')
                for v in ('high', 'full') for m in A_MODELS]
    if name == 'b':
        return [(f'b_{m}', ['b_propagation.py', '--model', m], f'data/b_{m}.json') for m in B_MODELS]
    if name == 'd':
        return [(f'd_{m}_{iv}', ['d_intervention.py', '--model', m, '--intervene', iv], f'data/d_{m}_{iv}.json')
                for m in ('spanetv2_s18_pure', 'spanetv2_s18_hybrid') for iv in ('none', 'flat')]
    raise ValueError(name)


def log(msg):
    print(f'[{time.strftime("%F %T")}] {msg}', flush=True)


def worker(gpu, q):
    while True:
        try:
            name, argv, out = q.get_nowait()
        except queue.Empty:
            return
        lock = os.path.join(HERE, 'logs', f'{name}.running')
        if os.path.exists(os.path.join(HERE, out)) or os.path.exists(lock):
            continue
        open(lock, 'w').write(str(gpu))
        log(f'GPU{gpu} start {name}')
        t0 = time.time()
        with open(os.path.join(HERE, 'logs', f'{name}.log'), 'w') as f:
            r = subprocess.run([PY, '-W', 'ignore', *argv], cwd=HERE, stdout=f, stderr=subprocess.STDOUT,
                               env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu)})
        os.remove(lock)
        log(f'GPU{gpu} end   {name} exit={r.returncode} {time.time() - t0:.0f}s')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpus', nargs='+', type=int, default=[0, 1, 2, 3])
    ap.add_argument('--list', required=True)
    args = ap.parse_args()
    q = queue.Queue()
    for j in jobs(args.list):
        q.put(j)
    ts = [threading.Thread(target=worker, args=(g, q)) for g in args.gpus]
    for t in ts:
        t.start()
        time.sleep(3)
    for t in ts:
        t.join()
    log('ALL DONE')


if __name__ == '__main__':
    main()
