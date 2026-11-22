#!/usr/bin/env python
"""Write conclusions.md (<= 3 sentences per experiment, numbers only) from data/raw; read by make_report.py."""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, 'data', 'raw')
M = ['spanetv2_s18_pure', 'spanetv2_s18_hybrid', 'convformer_s18', 'caformer_s18', 'dfformer_s18', 'cdfformer_s18',
     'convnext_tiny', 'swin_tiny', 'deit_small']
N = dict(zip(M, ['SPANetV2-S18-pure', 'SPANetV2-S18-hybrid', 'ConvFormer-S18', 'CAFormer-S18', 'DFFormer-S18',
                 'CDFFormer-S18', 'ConvNeXt-T', 'Swin-T', 'DeiT-S']))
P, Hy = 'spanetv2_s18_pure', 'spanetv2_s18_hybrid'


def j(name):
    return json.load(open(os.path.join(R, name)))


def rank(d, k, high_is_better=True):
    return sorted(d, key=lambda m: -d[m] if high_is_better else d[m]).index(k) + 1


def best(d, high_is_better=True):
    k = max(d, key=d.get) if high_is_better else min(d, key=d.get)
    return N[k], d[k]


def worst(d, high_is_better=True):
    return best(d, not high_is_better)


def main():
    out = []
    # A
    band = {m: j(f'band_{m}.json')['top1'] for m in M}
    lo = {m: np.mean([band[m][f'low{S}'] for S in range(20, 221, 20)]) for m in M}
    hi = {m: np.mean([band[m][f'high{S}'] for S in range(170, 221, 10)]) for m in M}
    h220 = {m: band[m]['high220'] for m in M}
    out += ['## A',
            f'저역 통과 11개 크기(S = 20–220)의 평균 top-1은 {N[Hy]} {lo[Hy]:.2f}(9개 중 {rank(lo, Hy)}위), {N[P]} {lo[P]:.2f}'
            f'({rank(lo, P)}위)이고, 가장 높은 값은 {best(lo)[0]} {best(lo)[1]:.2f}, 가장 낮은 값은 {worst(lo)[0]} {worst(lo)[1]:.2f}이다. '
            f'고역 통과 6개 크기(S = 170–220)의 평균 top-1은 {N[Hy]} {hi[Hy]:.2f}({rank(hi, Hy)}위), {N[P]} {hi[P]:.2f}'
            f'({rank(hi, P)}위)이고, 가장 높은 값은 {best(hi)[0]} {best(hi)[1]:.2f}, 가장 낮은 값은 {worst(hi)[0]} {worst(hi)[1]:.2f}이다. '
            f'고역 통과 S = 220에서 {N[P]}는 {h220[P]:.2f}로 9개 중 {rank(h220, P)}위, {N[Hy]}는 {h220[Hy]:.2f}로 '
            f'{rank(h220, Hy)}위이다.', '']
    # B
    low, high, clean = {}, {}, {}
    for m in M:
        g = np.load(os.path.join(R, f'fourier_{m}.npy')) * 100
        H = g.shape[0] // 2
        u, v = np.meshgrid(np.arange(-H, H + 1), np.arange(-H, H + 1), indexing='ij')
        low[m] = float(np.nanmean(g[np.sqrt(u ** 2 + v ** 2) / 112 <= 0.25 + 1e-9]))
        pr = np.array([(np.hypot(*map(int, k.split(','))) / 112, e * 100)
                       for k, e in j(f'fourier_sparse_{m}.json')['error'].items()])
        high[m] = float(pr[pr[:, 0] > 0.5, 1].mean())
        clean[m] = 100 - j(f'clean_{m}.json')['subset5k_top1']
    out += ['## B',
            f'5,000장 Fourier heat map에서 저역(ρ ≤ 0.25) 평균 오류율은 {N[Hy]} {low[Hy]:.2f}%(9개 중 낮은 순 {rank(low, Hy, False)}위), '
            f'{N[P]} {low[P]:.2f}%({rank(low, P, False)}위)이고, 가장 높은 값은 {worst(low, False)[0]} {worst(low, False)[1]:.2f}%이다. '
            f'고역(ρ > 0.5, stride 8 격자) 평균 오류율은 {N[Hy]} {high[Hy]:.2f}%({rank(high, Hy, False)}위), {N[P]} {high[P]:.2f}%'
            f'({rank(high, P, False)}위)이고, 가장 높은 값은 {worst(high, False)[0]} {worst(high, False)[1]:.2f}%이다. '
            f'같은 5,000장의 잡음 없는 오류율은 {min(clean.values()):.2f}–{max(clean.values()):.2f}%이다'
            f'({N[Hy]} {clean[Hy]:.2f}%, {N[P]} {clean[P]:.2f}%).', '']
    # C
    pg = {m: j(f'pgd_{m}.json') for m in M}
    t = lambda v, e: {m: pg[m]['results'][f'{v}|{e}']['top1'] for m in M}
    f1, l1, h1 = t('full', '1.0'), t('low', '1.0'), t('high', '1.0')
    leak = [pg[m]['results'][f'{v}|1.0']['out_of_band_energy_mean'] * 100 for m in M for v in ('low', 'high')]
    out += ['## C',
            f'제한 없는 L2 PGD ε = 1.0에서 top-1은 {N[Hy]} {f1[Hy]:.2f}%(9개 중 {rank(f1, Hy)}위), {N[P]} {f1[P]:.2f}%'
            f'({rank(f1, P)}위)이고, 가장 낮은 값은 {worst(f1)[0]} {worst(f1)[1]:.2f}%이다. '
            f'ε = 1.0에서 저역만 제한한 공격(M_l^56)은 {N[Hy]} {l1[Hy]:.2f}%({rank(l1, Hy)}위; 1위 {best(l1)[0]} {best(l1)[1]:.2f}%), '
            f'{N[P]} {l1[P]:.2f}%({rank(l1, P)}위)이고, 고역만 제한한 공격은 {N[Hy]} {h1[Hy]:.2f}%({rank(h1, Hy)}위), '
            f'{N[P]} {h1[P]:.2f}%({rank(h1, P)}위)이다. '
            f'마지막 [0,1] clamp 뒤 δ 에너지 가운데 대역 밖 비율의 평균은 ε = 1.0에서 모델·변형마다 {min(leak):.2f}–{max(leak):.2f}%이다.', '']
    # D1 / D2
    seg = {}
    for p in os.listdir(R):
        if p.startswith('seg_'):
            d = json.load(open(os.path.join(R, p)))
            b = j(f'boundary_{d["model"]}.json')
            seg[d['model']] = (d['name'], d['setting'], d['pair'], b['mIoU'], b['boundary_IoU'])
    up = {k: v for k, v in seg.items() if v[1] == 'upernet'}
    fmt = lambda keys: ', '.join(f'{seg[k][0]} {seg[k][3]:.2f} / {seg[k][4]:.2f}' for k in keys)
    o_m = sorted(up, key=lambda k: -up[k][3])
    o_b = sorted(up, key=lambda k: -up[k][4])
    out += ['## D1',
            f'UperNet 단일 스케일의 mIoU / Boundary IoU는 S18 짝에서 '
            + fmt(['upernet_spanetv2_s18_pure', 'upernet_spanetv2_s18_hybrid', 'upernet_convnext_tiny', 'upernet_swin_tiny'])
            + '이고, S36 짝에서 '
            + fmt(['upernet_spanetv2_s36_pure', 'upernet_spanetv2_s36_hybrid', 'upernet_convnext_small', 'upernet_swin_small'])
            + '이다. ' + ('여덟 모델의 mIoU 순위와 Boundary IoU 순위는 같다.' if o_m == o_b else
                          f'여덟 모델의 mIoU 순위와 Boundary IoU 순위는 다르다(mIoU: {", ".join(seg[k][0] for k in o_m)}; '
                          f'Boundary IoU: {", ".join(seg[k][0] for k in o_b)}).'), '']
    pairs = [('fpn_spanetv2_s18_pure', 'fpn_poolformer_s24'), ('fpn_spanetv2_s18_hybrid', 'fpn_poolformer_s24'),
             ('fpn_spanetv2_s36_pure', 'fpn_poolformer_s36'), ('fpn_spanetv2_s36_hybrid', 'fpn_poolformer_s36')]
    dm = [seg[a][3] - seg[b][3] for a, b in pairs]
    db = [seg[a][4] - seg[b][4] for a, b in pairs]
    out += ['## D2',
            'Semantic FPN의 mIoU / Boundary IoU는 '
            + fmt(['fpn_spanetv2_s18_pure', 'fpn_spanetv2_s18_hybrid', 'fpn_poolformer_s24', 'fpn_spanetv2_s36_pure',
                   'fpn_spanetv2_s36_hybrid', 'fpn_poolformer_s36']) + '이다. '
            f'같은 크기 짝(S18 ↔ PoolFormer-S24, S36 ↔ PoolFormer-S36)에서 SPANetV2와 PoolFormer의 차이는 '
            f'mIoU {min(dm):.2f}–{max(dm):.2f}, Boundary IoU {min(db):.2f}–{max(db):.2f}이다.', '']
    # D3
    det = {}
    for p in os.listdir(R):
        if p.startswith('det_'):
            d = json.load(open(os.path.join(R, p)))
            b = j(f'boundary_ap_{d["model"]}.json')
            det[d['model']] = (d['name'], d['metrics']['segm_mAP'] * 100, b['boundary']['AP'])
    fd = lambda keys: ', '.join(f'{det[k][0]} {det[k][1]:.1f} / {det[k][2]:.2f}' for k in keys)
    out += ['## D3',
            'COCO val2017의 mask AP / Boundary AP는 Mask R-CNN 3×(S18 짝)에서 '
            + fd(['mask_rcnn_spanetv2_s18_pure', 'mask_rcnn_spanetv2_s18_hybrid', 'mask_rcnn_convnext_tiny',
                  'mask_rcnn_swin_tiny']) + '이다. '
            'Cascade Mask R-CNN 3×에서는 S18 짝 '
            + fd(['cascade_spanetv2_s18_pure', 'cascade_spanetv2_s18_hybrid', 'cascade_convnext_tiny', 'cascade_swin_tiny'])
            + ', S36 짝 '
            + fd(['cascade_spanetv2_s36_pure', 'cascade_spanetv2_s36_hybrid', 'cascade_convnext_small', 'cascade_swin_small'])
            + '이다.', '']
    open(os.path.join(HERE, 'conclusions.md'), 'w').write('\n'.join(out))
    print('\n'.join(out))


if __name__ == '__main__':
    main()
