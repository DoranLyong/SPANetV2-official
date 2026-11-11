"""
Compare SPANetV2 384 fine-tuning results with MetaFormer's ConvFormer / CAFormer 384 models.

    python compare_384.py                 # reads ./output/train
    python compare_384.py --out /path/to/output/train

Pairing: *_pure <-> ConvFormer, *_hybrid <-> CAFormer (same size).
SPANetV2 Top-1 is the best `eval_top1` in each run's summary.csv. With --model-ema that is
the EMA model's accuracy, so evaluate the matching checkpoint with `validate.py --use-ema`.
If a model has several 384 runs, the most recent one is used.
"""
import argparse
import csv
import glob
import os

# MetaFormer README, ImageNet-1K only (224 pretrained -> 384 fine-tuned; crop_pct 1.0, bicubic).
# name: (params M, MACs G, top-1 %)
METAFORMER_384 = {
    'ConvFormer-S18': (27, 11.6, 84.4),
    'CAFormer-S18':   (26, 13.4, 85.0),
    'ConvFormer-S36': (40, 22.4, 85.4),
    'CAFormer-S36':   (39, 26.0, 85.7),
}

# SPANetV2 at 384: fvcore MACs, FFT not counted (FLOPs_count.ipynb "simple way"; FFT adds < 1%).
# model: (baseline, params M, MACs G)
SPANETV2_384 = {
    'spanetv2_s18_pure':   ('ConvFormer-S18', 29.2, 12.31),
    'spanetv2_s18_hybrid': ('CAFormer-S18',   27.0, 13.62),
    'spanetv2_s36_pure':   ('ConvFormer-S36', 44.1, 23.93),
    'spanetv2_s36_hybrid': ('CAFormer-S36',   40.7, 26.59),
}
EPOCHS = 30


def read_run(out_dir, model):
    runs = sorted(glob.glob(os.path.join(out_dir, f'*-{model}-384')))  # names start with a timestamp
    if not runs:
        return None
    path = os.path.join(runs[-1], 'summary.csv')
    if not os.path.isfile(path):
        return {'run': runs[-1], 'epochs': 0}
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {'run': runs[-1], 'epochs': 0}
    best = max(rows, key=lambda r: float(r['eval_top1']))
    return {'run': runs[-1], 'epochs': len(rows), 'best_epoch': int(best['epoch']),
            'top1': float(best['eval_top1']), 'top5': float(best['eval_top5'])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='./output/train', help='train.py output directory')
    args = parser.parse_args()

    header = f"| {'Model':20s} | {'Params':>6s} | {'MACs':>6s} | {'Top-1':>6s} | {'Δ':>5s} | Status"
    print(header)
    print('|' + '|'.join('-' * (len(c) + 0) for c in header.split('|')[1:]))
    for model, (base, params, macs) in SPANETV2_384.items():
        b_params, b_macs, b_top1 = METAFORMER_384[base]
        print(f"| {base:20s} | {b_params:5.1f}M | {b_macs:5.1f}G | {b_top1:6.1f} | {'':5s} | MetaFormer README")

        r = read_run(args.out, model)
        if r is None:
            top1, delta, status = '-', '', 'not started'
        elif r['epochs'] == 0:
            top1, delta, status = '-', '', 'running (no epoch done yet)'
        else:
            top1 = f"{r['top1']:6.2f}"
            delta = f"{r['top1'] - b_top1:+5.2f}"
            done = 'done' if r['epochs'] >= EPOCHS else 'running'
            status = f"{done} {r['epochs']}/{EPOCHS} ep, best @ ep {r['best_epoch']} (EMA)"
        size, kind = model.split('_')[1:]
        name = f'SPANetV2-{size.upper()}-{kind}'
        print(f"| {name:20s} | {params:5.1f}M | {macs:5.1f}G | {top1:>6s} | {delta:>5s} | {status}")


if __name__ == '__main__':
    main()
