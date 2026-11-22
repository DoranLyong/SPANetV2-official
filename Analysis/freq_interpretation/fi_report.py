#!/usr/bin/env python
"""Tables (data/*.csv), figures (figs/*.png, text >= 16 pt) and report.md from data/ (conclusions from conclusions.md).

    python fi_report.py
"""
import csv
import glob
import json
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')
FIGS = os.path.join(HERE, 'figs')
INK, INK2, MUTED, GRID = '#0b0b0b', '#52514e', '#898781', '#e1e0d9'
NAME = {'spanetv2_s18_pure': 'SPANetV2-S18-pure', 'spanetv2_s18_hybrid': 'SPANetV2-S18-hybrid',
        'convformer_s18': 'ConvFormer-S18', 'caformer_s18': 'CAFormer-S18', 'dfformer_s18': 'DFFormer-S18',
        'cdfformer_s18': 'CDFFormer-S18', 'deit_small': 'DeiT-S', 'spanetv2_s36_pure': 'SPANetV2-S36-pure',
        'spanetv2_s36_hybrid': 'SPANetV2-S36-hybrid'}
STY = {'spanetv2_s18_pure': ('#2a78d6', '-', 'o'), 'spanetv2_s18_hybrid': ('#2a78d6', '--', 's'),
       'convformer_s18': ('#eb6834', '-', 'o'), 'caformer_s18': ('#eb6834', '--', 's'),
       'dfformer_s18': ('#1baf7a', '-', 'o'), 'cdfformer_s18': ('#1baf7a', '--', 's'), 'deit_small': ('#008300', '-', 'D')}
A_MODELS = ['spanetv2_s18_hybrid', 'caformer_s18', 'spanetv2_s18_pure', 'convformer_s18', 'deit_small']
B_MODELS = ['spanetv2_s18_pure', 'convformer_s18', 'spanetv2_s18_hybrid', 'caformer_s18', 'dfformer_s18', 'cdfformer_s18']

plt.rcParams.update({
    'font.size': 16, 'axes.titlesize': 17, 'axes.labelsize': 16, 'xtick.labelsize': 16, 'ytick.labelsize': 16,
    'legend.fontsize': 16, 'figure.titlesize': 19, 'axes.edgecolor': '#c3c2b7', 'axes.labelcolor': INK2,
    'xtick.color': INK2, 'ytick.color': INK2, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8,
    'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 110, 'font.family': 'DejaVu Sans',
})


def jl(name):
    p = os.path.join(DATA, name)
    return json.load(open(p)) if os.path.exists(p) else None


def rcsv(name):
    p = os.path.join(DATA, name)
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []


def wcsv(name, header, rows):
    with open(os.path.join(DATA, name), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def CMP(key):
    k = key
    for p in ('upernet_', 'fpn_', 'mask_rcnn_', 'cascade_'):
        k = k.replace(p, '')
    return {'convnext_tiny': 'ConvNeXt-T', 'convnext_small': 'ConvNeXt-S', 'swin_tiny': 'Swin-T', 'swin_small': 'Swin-S',
            'poolformer_s24': 'PoolFormer-S24', 'poolformer_s36': 'PoolFormer-S36'}.get(k, k)


def table(header, rows):
    esc = lambda c: str(c).replace('|', '\\|')
    return '\n'.join(['| ' + ' | '.join(esc(h) for h in header) + ' |', '|' + '|'.join('---' for _ in header) + '|'] +
                     ['| ' + ' | '.join(esc(c) for c in r) + ' |' for r in rows])


def legend(fig, keys, ncol=5):
    h = [Line2D([], [], color=STY[k][0], ls=STY[k][1], marker=STY[k][2], ms=8, lw=2.2) for k in keys]
    fig.legend(h, [NAME[k] for k in keys], loc='outside lower center', ncol=ncol, frameon=False)


# ================================================================================================ A
def sec_a():
    L = ['## A. PGD 결과의 신뢰성 점검', '']
    rows, cfgs = [], []
    res = {}
    for m in A_MODELS:
        c = jl(f'a_clean_{m}.json')
        res[(m, 'clean')] = c['top1'] if c else None
        for v in ('high', 'full'):
            d = jl(f'a_main_{m}_{v}.json')
            if d:
                for k, r in d['results'].items():
                    if 'nondeterministic' in k:
                        res[(m, f'{v}|1.0|10|repeat')] = r['top1']
                    else:
                        res[(m, f'{v}|{k}')] = r['top1']
        d = jl(f'a_apgd_{m}.json')
        if d:
            for e, r in d['results'].items():
                res[(m, f'full|{e}|apgd')] = r['top1']
        d = jl(f'a_large_{m}.json')
        if d:
            for e, r in d['results'].items():
                res[(m, f'full|{e}|100|norand')] = r['top1']
        for v in ('high', 'full'):
            d = jl(f'a_apgd2_{m}_{v}.json')
            if d:
                for e, r in d['results'].items():
                    res[(m, f'{v}|{e}|apgd')] = r['top1']
                    res[(m, f'{v}|{e}|apgd|oob')] = r['out_of_band_energy_share']
    order = ['clean']
    for v in ('high', 'full'):
        order.append(f'{v}|0.25|apgd')
        for e in ('0.5', '1.0'):
            order += [f'{v}|{e}|10|norand', f'{v}|{e}|50|norand', f'{v}|{e}|100|norand', f'{v}|{e}|100|rand5', f'{v}|{e}|apgd']
    order += [f'full|{e}|100|norand' for e in ('2.0', '4.0', '8.0')]
    out_rows = []
    for k in order:
        row = [k]
        for m in A_MODELS:
            v = res.get((m, k))
            row.append('—' if v is None else f'{v:.1f}')
            if v is not None:
                out_rows.append((m, k, f'{v:.4f}'))
        rows.append(row)
    wcsv('a_top1.csv', ['model', 'setting', 'top1'], out_rows)
    L += ['top-1 (%), 1,000장. 설정 = 변형 | ε | step | 시작(norand = 무작위 시작 없음, rand5 = 무작위 시작 5회 중 최악). '
          'apgd = APGD-CE(L2, 100 iter, 무작위 시작 1회; full은 autoattack 원본, high는 normalize만 대역 사영으로 바꾼 subclass).', '',
          table(['설정', *[NAME[m] for m in A_MODELS]], rows), '']
    rep = []
    for m in A_MODELS:
        d = jl(f'a_main_{m}_high.json') or {}
        t = d.get('results', {}).get('1.0|10|norand|nondeterministic_repeats', {}).get('top1', [])
        t = [f'{x:.1f}' for x in t] + ['—'] * (2 - len(t))
        # files written before the DeiT exception was added carry no flag; those runs used deterministic cuDNN
        det = d.get('deterministic_cudnn', True) if d else None
        rep.append([NAME[m], res.get((m, 'high|1.0|10|norand')) is not None and f"{res[(m, 'high|1.0|10|norand')]:.1f}" or '—',
                    *t, '—' if det is None else ('예' if det else '아니오')])
    L += ['같은 10-step 공격(high, ε = 1.0)을 비결정적 GPU 커널로 두 번 더 돌린 top-1. "본 실행"은 위 표의 값이고, 그 실행이 '
          '결정적 cuDNN이었는지를 마지막 열에 적었다(DeiT-S는 결정적 역전파가 약 10배 느려 비결정적으로 돌렸다):', '',
          table(['모델', '본 실행', '반복 1', '반복 2', '본 실행 결정적'], rep), '']
    # CE convergence: top-1 50 -> 100 steps, CE at 50 / 100, share of the CE rise that happens in the last 10 steps
    crow = []
    for v in ('high', 'full'):
        for e in ('0.5', '1.0'):
            for m in A_MODELS:
                d = jl(f'a_main_{m}_{v}.json')
                if not d or f'{e}|100|norand' not in d['results']:
                    continue
                lc = np.array(d['results'][f'{e}|100|norand']['loss_curve'])
                t50, t100 = res.get((m, f'{v}|{e}|50|norand')), res.get((m, f'{v}|{e}|100|norand'))
                crow.append([v, e, NAME[m], f'{t50:.1f}', f'{t100:.1f}', f'{t100 - t50:+.1f}', f'{lc[0]:.2f}', f'{lc[50]:.2f}',
                             f'{lc[100]:.2f}', f'{(lc[100] - lc[90]) / (lc[100] - lc[0]):.3f}'])
    if crow:
        wcsv('a_ce_convergence.csv', ['variant', 'eps', 'model', 'top1_50', 'top1_100', 'diff', 'ce_0', 'ce_50', 'ce_100',
                                      'last10_share_of_rise'], crow)
        L += ['CE 수렴(100 step, 무작위 시작 없음). 평균 CE는 이미 틀린 영상에서도 계속 오르므로(상한 없음), 수렴은 top-1의 '
              '50 → 100 step 변화와 마지막 10 step에서 오른 CE가 전체 상승분에서 차지하는 비율로 본다(선형 증가면 0.10):', '',
              table(['변형', 'ε', '모델', 'top-1 50', 'top-1 100', '차이', 'CE 0', 'CE 50', 'CE 100', '마지막 10 step 비율'], crow), '']
    # paired differences on the same 1,000 images (per-image correctness stored by part_main)
    corr = {}
    for m in A_MODELS:
        for v in ('high', 'full'):
            d = jl(f'a_main_{m}_{v}.json')
            if d:
                for k, r in d['results'].items():
                    if 'correct' in r:
                        corr[(m, f'{v}|{k}')] = np.array(r['correct'], bool)
            d = jl(f'a_apgd2_{m}_{v}.json')
            if d:
                for e, r in d['results'].items():
                    corr[(m, f'{v}|{e}|apgd')] = np.array(r['correct'], bool)
    rng = np.random.default_rng(0)
    prow = []
    for a, b in (('spanetv2_s18_hybrid', 'caformer_s18'), ('spanetv2_s18_pure', 'convformer_s18'),
                 ('spanetv2_s18_hybrid', 'deit_small')):
        for k in order:
            if (a, k) not in corr or (b, k) not in corr:
                continue
            ca, cb = corr[(a, k)], corr[(b, k)]
            diff = ca.astype(float) - cb.astype(float)
            boot = diff[rng.integers(0, len(diff), (5000, len(diff)))].mean(1) * 100
            lo, hi = np.percentile(boot, [2.5, 97.5])
            prow.append([f'{NAME[a]} − {NAME[b]}', k, f'{diff.mean() * 100:+.1f}', f'[{lo:+.1f}, {hi:+.1f}]',
                         int((ca & ~cb).sum()), int((~ca & cb).sum())])
    if prow:
        wcsv('a_paired.csv', ['pair', 'setting', 'diff_top1', 'ci95_bootstrap', 'only_first_correct', 'only_second_correct'], prow)
        L += ['같은 1,000장에서의 짝 차이(top-1 차이, 영상 단위 bootstrap 5,000회 95% 구간, 한쪽만 맞힌 영상 수):', '',
              table(['짝', '설정', '차이', '95% 구간', '앞 모델만 맞힘', '뒤 모델만 맞힘'], prow), '']
    tr = jl('a_transfer.json')
    if tr:
        trows = []
        for k, r in tr['results'].items():
            src, rest = k.split('->')
            tgt, v = rest.split('|')
            own = res.get((tgt, f'{v}|1.0|100|norand'))
            trows.append([NAME[src], NAME[tgt], v, f"{r['top1']:.1f}", '—' if own is None else f'{own:.1f}'])
        L += ['전이 공격(ε = 1.0, 100 step, 무작위 시작 없음): 원천 모델에서 만든 δ를 대상 모델에 넣은 top-1과, 대상 모델 자신의 백색 공격 top-1:', '',
              table(['원천', '대상', '변형', '전이 top-1', '대상의 백색 top-1'], trows), '']
    # figure: top-1 vs attack strength
    fig, axes = plt.subplots(1, 4, figsize=(24, 6.8), constrained_layout=True, sharey=True)
    for j, (v, e) in enumerate([('high', '0.5'), ('high', '1.0'), ('full', '0.5'), ('full', '1.0')]):
        xs = ['10', '50', '100', '100\n+5 rs', 'APGD']
        keys = [f'{v}|{e}|10|norand', f'{v}|{e}|50|norand', f'{v}|{e}|100|norand', f'{v}|{e}|100|rand5', f'{v}|{e}|apgd']
        for m in A_MODELS:
            y = [res.get((m, k)) for k in keys]
            axes[j].plot(range(len(xs)), [np.nan if t is None else t for t in y], color=STY[m][0], ls=STY[m][1],
                         marker=STY[m][2], ms=8, lw=2.2)
        axes[j].set_xticks(range(len(xs)))
        axes[j].set_xticklabels(xs)
        axes[j].set_title(f'{v} band, ε = {e}' if v == 'high' else f'unrestricted, ε = {e}')
        axes[j].set_xlabel('attack (steps / restarts)')
    axes[0].set_ylabel('top-1 (%), 1,000 images')
    legend(fig, A_MODELS)
    fig.savefig(os.path.join(FIGS, 'a_attack_strength.png'), bbox_inches='tight')
    plt.close(fig)
    # loss curves
    fig, axes = plt.subplots(1, 4, figsize=(24, 6.8), constrained_layout=True)
    for j, (v, e) in enumerate([('high', '0.5'), ('high', '1.0'), ('full', '0.5'), ('full', '1.0')]):
        for m in A_MODELS:
            d = jl(f'a_main_{m}_{v}.json')
            if d and f'{e}|100|norand' in d['results']:
                lc = d['results'][f'{e}|100|norand']['loss_curve']
                axes[j].plot(range(len(lc)), lc, color=STY[m][0], ls=STY[m][1], lw=2.2)
        axes[j].set_title(f'{v}, ε = {e}, 100 steps')
        axes[j].set_xlabel('step')
    axes[0].set_ylabel('mean cross-entropy')
    legend(fig, A_MODELS)
    fig.savefig(os.path.join(FIGS, 'a_loss_curves.png'), bbox_inches='tight')
    plt.close(fig)
    # large eps
    fig, ax = plt.subplots(figsize=(11, 6.8), constrained_layout=True)
    for m in A_MODELS:
        e = ['0.5', '1.0', '2.0', '4.0', '8.0']
        y = [res.get((m, f'full|{x}|100|norand')) for x in e]
        ax.plot([float(x) for x in e], [np.nan if t is None else t for t in y], color=STY[m][0], ls=STY[m][1],
                marker=STY[m][2], ms=8, lw=2.2)
    ax.set_xscale('log', base=2)
    ax.set(xlabel='ε (L2, unrestricted, 100 steps)', ylabel='top-1 (%)')
    legend(fig, A_MODELS, ncol=3)
    fig.savefig(os.path.join(FIGS, 'a_large_eps.png'), bbox_inches='tight')
    plt.close(fig)
    L += ['![attack strength](figs/a_attack_strength.png)', '', '![loss curves](figs/a_loss_curves.png)', '',
          '![large eps](figs/a_large_eps.png)', '']
    return L, res


# ================================================================================================ E
def mean_by(rows, keys, val, fn=np.mean):
    d = defaultdict(list)
    for r in rows:
        try:
            x = float(r[val])
        except ValueError:
            continue
        if np.isfinite(x):
            d[tuple(r[k] for k in keys)].append(x)
    return {k: fn(v) for k, v in d.items()}


def sec_e():
    L = ['## E. SPAM 설계 의도의 검증', '']
    r1 = rcsv('e1_kernel_bands.csv')
    order = mean_by([r for r in r1 if r['kernel'] == '3'], ['arch', 'stage'], 'centroids_in_kernel_order')
    pair = mean_by([r for r in r1 if r['kernel'] == '3'], ['arch', 'stage'], 'pairwise_order_fraction')
    cen = mean_by(r1, ['arch', 'stage', 'kernel'], 'centroid_rho')
    shares = {b: mean_by(r1, ['arch', 'stage', 'kernel'], f'share_{b}') for b in ('low', 'mid', 'high')}
    rows = []
    for (a, s), v in sorted(order.items()):
        rows.append([NAME[a], s, f'{v:.3f}', f'{pair[(a, s)]:.3f}'] + [f"{cen[(a, s, k)]:.3f}" for k in '3579'] +
                    [f"{shares['high'][(a, s, k)]:.3f}" for k in '3579'])
    L += ['### E1. 커널이 대역을 나누는가', '',
          '블록·채널 번호마다 네 head(k = 3, 5, 7, 9)의 |K̂|² 무게중심 ρ̄를 구해, 커널 크기 순서(k3 > k5 > k7 > k9)와 맞는지 셌다. '
          '"네 개 모두 순서" 확률은 무작위면 1/24 = 0.042, "쌍별 순서"는 0.5다.', '',
          table(['모델', '단계', '네 개 모두 순서', '쌍별 순서', 'ρ̄ k3', 'ρ̄ k5', 'ρ̄ k7', 'ρ̄ k9', '고역 비중 k3', 'k5', 'k7', 'k9'], rows), '']
    r2 = rcsv('e2_srf_vs_kernel.csv')
    keys = ['arch', 'stage']
    vals = {k: mean_by(r2, keys, k) for k in ('rho_abs', 'rho_rel', 'otherhead_abs_mean', 'otherhead_rel_mean',
                                               'shuffle_abs', 'shuffle_rel', 'frac_self_gt_otherhead_abs',
                                               'frac_self_gt_otherhead_rel', 'self_gt_shuffle_abs', 'self_gt_shuffle_rel')}
    pos = {k: mean_by([dict(r, pa=str(float(float(r['rho_abs']) > 0)), pr=str(float(float(r['rho_rel']) > 0))) for r in r2],
                      keys, k) for k in ('pa', 'pr')}
    rows = []
    for k in sorted(vals['rho_abs']):
        rows.append([NAME[k[0]], k[1], f"{vals['rho_abs'][k]:+.3f}", f"{pos['pa'][k]:.2f}", f"{vals['otherhead_abs_mean'][k]:+.3f}",
                     f"{vals['shuffle_abs'][k]:+.3f}", f"{vals['frac_self_gt_otherhead_abs'][k]:.2f}", f"{vals['self_gt_shuffle_abs'][k]:.2f}",
                     f"{vals['rho_rel'][k]:+.3f}", f"{pos['pr'][k]:.2f}", f"{vals['otherhead_rel_mean'][k]:+.3f}",
                     f"{vals['shuffle_rel'][k]:+.3f}", f"{vals['frac_self_gt_otherhead_rel'][k]:.2f}", f"{vals['self_gt_shuffle_rel'][k]:.2f}"])
    L += ['### E2. SRF가 부족한 대역을 보정하는가 (파라미터)', '',
          '채널마다 2D 격자의 서로 다른 주파수 전체(DC 제외)에서 Spearman 상관을 구해 평균했다. (a)는 log ψ_h 대 −log|K̂_h|이고, '
          '(b)는 log ψ_h 대 −log(|K̃_h| / head 평균 |K̃|)이다. 대조군은 다른 head 커널과 짝지은 상관(3개 평균)과 채널 순서를 섞어 짝지은 상관이다.', '',
          table(['모델', '단계', '(a) 평균', '(a) >0 비율', '(a) 다른 head', '(a) 섞은 채널', '(a) 자기>다른 head', '(a) 자기>섞은',
                 '(b) 평균', '(b) >0 비율', '(b) 다른 head', '(b) 섞은 채널', '(b) 자기>다른 head', '(b) 자기>섞은'], rows), '']
    kk = mean_by(r2, ['arch', 'stage', 'kernel'], 'rho_abs')
    kr = mean_by(r2, ['arch', 'stage', 'kernel'], 'rho_rel')
    rows = [[NAME[a], s] + [f"{kk[(a, s, k)]:+.3f} / {kr[(a, s, k)]:+.3f}" for k in '3579']
            for (a, s) in sorted({(x[0], x[1]) for x in kk})]
    L += ['커널 크기별 평균 상관 (a) / (b):', '', table(['모델', '단계', 'k3', 'k5', 'k7', 'k9'], rows), '']
    r3 = rcsv('e3_flatness.csv')
    cv = mean_by(r3, ['arch', 'stage', 'response'], 'cv_across_bands')
    hl = mean_by(r3, ['arch', 'stage', 'response'], 'high_over_low')
    rows = []
    for a in ['spanetv2_s18_pure', 'spanetv2_s18_hybrid', 'spanetv2_s36_pure', 'spanetv2_s36_hybrid']:
        for s in '1234':
            if (a, s, 'kernel_only') in cv:
                dec = np.mean([float(x['cv_across_bands']) for x in r3 if x['arch'] == a and x['stage'] == s and x['response'] == 'after_SRF']) - cv[(a, s, 'kernel_only')]
                rows.append([NAME[a], s, f"{cv[(a, s, 'kernel_only')]:.3f}", f"{cv[(a, s, 'after_SRF')]:.3f}", f'{dec:+.3f}',
                             f"{hl.get((a, s, 'kernel_only'), np.nan):.3f}", f"{hl.get((a, s, 'after_SRF'), np.nan):.3f}",
                             f"{cv.get(('convformer_s18', s, 'dwconv |K^|'), np.nan):.3f}",
                             f"{cv.get(('dfformer_s18', s, 'predicted |W| (image mean)'), np.nan):.3f}"])
    frac = {}
    by = defaultdict(dict)
    for x in r3:
        if x['response'] in ('kernel_only', 'after_SRF'):
            by[(x['arch'], x['stage'], x['block'], x['channel'])][x['response']] = float(x['cv_across_bands'])
    for (a, s, b, c), v in by.items():
        frac.setdefault((a, s), []).append(v['after_SRF'] < v['kernel_only'])
    for r in rows:
        a = [k for k, n in NAME.items() if n == r[0]][0]
        r.insert(5, f"{np.mean(frac[(a, r[1])]):.2f}")
    L += ['### E3. 보정이 mixer 전체 응답을 고르게 하는가', '',
          '채널 번호 c마다 네 head를 합친 Σ_h|K̂_h|(커널만)와 Σ_h ψ_h|K̂_h|(SRF 후)의 대역(저·중·고, DC 제외) 평균 사이 변동계수(CV, 작을수록 평탄)와 고/저 비다. '
          'ConvFormer-S18은 같은 단계 7×7 depthwise 커널의 |K̂|이고, DFFormer-S18은 1,000장 평균 예측 필터 |W|이다(채널마다 계산해 평균). 4단계(7×7)는 저역 칸이 없어 CV는 중·고역 둘로 계산했다.', '',
          table(['모델', '단계', 'CV 커널만', 'CV SRF 후', '차이', 'CV가 줄어든 채널 비율', '고/저 커널만', '고/저 SRF 후',
                 'CV ConvFormer', 'CV DFFormer'], rows), '']
    r4 = rcsv('e3_centroid_spread.csv')
    sp = defaultdict(list)
    for x in r4:
        sp[(x['arch'], x['stage'])].append((float(x['range_kernel']), float(x['range_after_SRF'])))
    rows = [[NAME[a], s, f'{np.mean([t[0] for t in v]):.3f}', f'{np.mean([t[1] for t in v]):.3f}',
             f'{np.mean([t[1] > t[0] for t in v]):.2f}'] for (a, s), v in sorted(sp.items())]
    L += ['head 사이 무게중심 ρ̄의 범위(max − min)는 커널만일 때와 SRF 후로 비교했다(넓어진 채널 비율):', '',
          table(['모델', '단계', '범위 커널만', '범위 SRF 후', '넓어진 비율'], rows), '']
    r5 = rcsv('e4_activation_gain.csv')
    rows = []
    if r5:
        g = lambda key: mean_by(r5, ['arch', 'stage'], key)
        ra, rr, ca, cr, fa, fr = (g(k) for k in ('rho_abs', 'rho_rel', 'otherhead_psi_rho_abs', 'otherhead_psi_rho_rel',
                                                  'frac_self_gt_otherhead_abs', 'frac_self_gt_otherhead_rel'))
        pa = mean_by([dict(r, p=str(float(float(r['rho_abs']) > 0))) for r in r5], ['arch', 'stage'], 'p')
        pr = mean_by([dict(r, p=str(float(float(r['rho_rel']) > 0))) for r in r5], ['arch', 'stage'], 'p')
        ua, ur = g('weak_band_share_up_abs'), g('weak_band_share_up_rel')
        for k in sorted(ra):
            rows.append([NAME[k[0]], k[1], f'{ra[k]:+.3f}', f'{pa[k]:.2f}', f'{ca[k]:+.3f}', f'{fa[k]:.2f}', f'{rr[k]:+.3f}',
                         f'{pr[k]:.2f}', f'{cr[k]:+.3f}', f'{fr[k]:.2f}', f'{ua[k]:.2f}', f'{ur[k]:.2f}'])
    L += ['### E4. 실제 영상에서 (IN1K val 1,000장, S18 두 모델)', '',
          'head마다 x2(convolution 후)와 x3(SRF 후)의 채널 평균 power P2, P3로 이득 log(P3/P2)를 만들었다. 이것을 (a) −log P2, (b) −log(P2_h / head 평균)과 주파수마다 상관시켰다(영상·블록·head마다 Spearman, 평균). '
          '대조군은 같은 conv 출력에 다른 head의 ψ를 적용했을 때의 이득과의 상관이다. "약한 대역 비중 증가"는 x2에서 가장 약한 대역((a) 비중 최소, (b) head 평균 대비 최소)의 비중이 x3에서 늘어난 (영상·블록·head) 비율이다.', '',
          table(['모델', '단계', '(a) 상관', '(a) >0 영상 비율', '(a) 다른 head ψ', '(a) 자기>다른', '(b) 상관', '(b) >0 영상 비율',
                 '(b) 다른 head ψ', '(b) 자기>다른', '약한 대역 비중↑ (a)', '약한 대역 비중↑ (b)'], rows), '']
    return L


# ================================================================================================ B
def sec_b():
    L = ['## B. 섭동이 네트워크 어디에서 커지는가', '']
    D = {m: jl(f'b_{m}.json') for m in B_MODELS}
    if not all(D.values()):
        return L + ['(결과 없음)', '']
    rows = []
    perts = {'Fourier 상위 5': [f'fourier_top_{i}' for i in range(5)], 'Fourier 하위 5': [f'fourier_bottom_{i}' for i in range(5)],
             'PGD 자기 δ': ['pgd_own'], 'PGD CAFormer δ': ['pgd_caformer']}
    out_rows = []
    for m in B_MODELS:
        for p in D[m]['points']:
            for pn, keys in perts.items():
                out_rows.append((m, p['kind'], p['stage'], p['block'], pn, f"{np.mean([p[k]['r_mean'] for k in keys]):.6g}",
                                 f"{np.mean([p[k]['high_share_mean'] for k in keys]):.6g}"))
    wcsv('b_propagation.csv', ['model', 'kind', 'stage', 'block', 'perturbation', 'r_mean', 'high_share_mean'], out_rows)
    for m in B_MODELS:
        rows.append([NAME[m], f"{D[m]['top1']['clean']:.1f}"] + [f"{np.mean([D[m]['top1'][k] for k in keys]):.1f}" for keys in perts.values()])
    L += ['top-1 (%), 1,000장:', '', table(['모델', 'clean', *perts], rows), '']
    srows = []
    for m in B_MODELS:
        st = [p for p in D[m]['points'] if p['kind'] == 'stage']
        stem = [p for p in D[m]['points'] if p['kind'] == 'stem'][0]
        for pn, keys in perts.items():
            srows.append([NAME[m], pn, f"{np.mean([stem[k]['r_mean'] for k in keys]):.3f}"] +
                         [f"{np.mean([p[k]['r_mean'] for k in keys]):.3f}" for p in st] +
                         [f"{np.mean([p[k]['high_share_mean'] for k in keys]):.2f}" for p in st])
    L += ['stem과 단계 출력의 상대 변화 r(영상 평균)과 Δh의 고역 에너지 비율:', '',
          table(['모델', '섭동', 'r stem', 'r 1단계', 'r 2단계', 'r 3단계', 'r 4단계', '고역비 1', '고역비 2', '고역비 3', '고역비 4'], srows), '']
    brow = []
    for m in B_MODELS:
        for pn, keys in perts.items():
            for s in (1, 2, 3, 4):
                mx = [np.mean([p[k]['r_mean'] for k in keys]) for p in D[m]['points'] if p['kind'] == 'mixer' and p['stage'] == s]
                ml = [np.mean([p[k]['r_mean'] for k in keys]) for p in D[m]['points'] if p['kind'] == 'mlp' and p['stage'] == s]
                brow.append((m, pn, s, np.mean(mx), np.mean(ml)))
    wcsv('b_branch_mean.csv', ['model', 'perturbation', 'stage', 'mixer_r_mean', 'mlp_r_mean'],
         [(a, b, c, f'{d:.6g}', f'{e:.6g}') for a, b, c, d, e in brow])
    rows = []
    for (a, b) in (('spanetv2_s18_pure', 'convformer_s18'), ('spanetv2_s18_hybrid', 'caformer_s18')):
        for pn in perts:
            for s in (1, 2, 3, 4):
                ra = [x for x in brow if x[0] == a and x[1] == pn and x[2] == s][0]
                rb = [x for x in brow if x[0] == b and x[1] == pn and x[2] == s][0]
                rows.append([f'{NAME[a]} / {NAME[b]}', pn, s, f'{ra[3]:.3f} / {rb[3]:.3f}', f'{ra[3] / rb[3]:.2f}',
                             f'{ra[4]:.3f} / {rb[4]:.3f}', f'{ra[4] / rb[4]:.2f}'])
    L += ['짝별 단계 평균 가지 출력 r(블록 평균): mixer 가지와 MLP 가지. 배율은 SPANetV2 / 비교 모델이다.', '',
          table(['짝', '섭동', '단계', 'mixer r', 'mixer 배율', 'MLP r', 'MLP 배율'], rows), '']
    # figure
    fig, axes = plt.subplots(2, 4, figsize=(26, 12), constrained_layout=True, sharey='col')
    for i, (a, b) in enumerate((('spanetv2_s18_pure', 'convformer_s18'), ('spanetv2_s18_hybrid', 'caformer_s18'))):
        for j, (pn, keys) in enumerate(perts.items()):
            ax = axes[i, j]
            for m in (a, b):
                pts = D[m]['points']
                xs = list(range(len(pts)))
                for kind, mk in (('mixer', 'o'), ('mlp', '^'), ('stage', 's')):
                    sel = [k for k, p in enumerate(pts) if p['kind'] == kind]
                    ax.plot(sel, [np.mean([pts[k][q]['r_mean'] for q in keys]) for k in sel], color=STY[m][0],
                            ls='' if kind != 'stage' else STY[m][1], marker=mk, ms=6 if kind != 'stage' else 9, alpha=0.85,
                            lw=2)
                for k, p in enumerate(pts):
                    if p['kind'] == 'stage':
                        ax.axvline(k + 0.5, color=MUTED, ls=':', lw=1)
            ax.set_yscale('log')
            ax.set_title({'Fourier 상위 5': 'Fourier top 5 (ConvFormer error)', 'Fourier 하위 5': 'Fourier bottom 5',
                          'PGD 자기 δ': 'PGD, own δ', 'PGD CAFormer δ': 'PGD, CAFormer δ'}[pn])
            if j == 0:
                ax.set_ylabel(f'{NAME[a]} vs {NAME[b]}\nr = ‖Δh‖ / ‖h‖')
            ax.set_xlabel('measurement point (stem → stage 4)')
    h = [Line2D([], [], color=STY[m][0], ls=STY[m][1], lw=2.2) for m in B_MODELS[:4]] + \
        [Line2D([], [], color=INK2, ls='', marker=mk, ms=9) for mk in 'o^s']
    fig.legend(h, [NAME[m] for m in B_MODELS[:4]] + ['token-mixer branch', 'MLP branch', 'stage output'],
               loc='outside lower center', ncol=7, frameon=False)
    fig.savefig(os.path.join(FIGS, 'b_propagation.png'), bbox_inches='tight')
    plt.close(fig)
    L += ['![propagation](figs/b_propagation.png)', '']
    return L


# ================================================================================================ C
def sec_c():
    L = ['## C. 경계 결과의 분해', '']
    g = rcsv('c_seg_gains.csv')
    m = rcsv('c_seg_metrics.csv')
    sub = jl('c_subsets.json')
    rows = []
    for r in g:
        rows.append([r['setting'], r['size'], NAME.get(r['spanetv2'].replace('upernet_', '').replace('fpn_', ''), r['spanetv2']),
                     CMP(r['comparison']), r['subset'], f"{float(r['gain_mIoU']):+.2f}",
                     f"{float(r['gain_boundary_IoU']):+.2f}", f"{float(r['gain_interior_IoU']):+.2f}"])
    L += [f"ADE20K의 interior IoU는 정답 경계 띠(대각선 2%) 밖의 화소만으로 계산했다(화소 비율 평균 {sub['interior_fraction_mean']:.3f}). "
          f"얇은 구조 상위 10%(200장)의 thin-structure score 경계값은 {sub['thin_threshold_top10']:.4f}이고, 전체 중앙값은 {sub['thin_median']:.4f}이다. "
          '이득은 SPANetV2 − 비교 모델이다.', '',
          table(['프레임워크', '크기', 'SPANetV2', '비교', '부분집합', 'mIoU 이득', 'Boundary IoU 이득', 'interior IoU 이득'], rows), '']
    mrows = [[x['model'], x['subset'], x['n_images'], f"{float(x['mIoU']):.2f}", f"{float(x['boundary_IoU']):.2f}",
              f"{float(x['interior_IoU']):.2f}"] for x in m]
    L += ['<details><summary>모델별 값</summary>', '', table(['모델', '부분집합', '영상 수', 'mIoU', 'Boundary IoU', 'interior IoU'], mrows), '', '</details>', '']
    d = rcsv('c_det_size.csv')
    rows = [[x['framework'], x['size'], NAME['spanetv2_' + x['spanetv2'].split('_spanetv2_')[1]], CMP(x['comparison']), x['metric'],
             f"{float(x['gain_boundary']):+.2f}", f"{float(x['gain_mask']):+.2f}"] for x in d]
    L += ['COCO val2017, 물체 크기별 이득(boundary-iou-api, Boundary AP와 같은 API의 mask AP):', '',
          table(['프레임워크', '크기', 'SPANetV2', '비교', '지표', 'Boundary AP 이득', 'mask AP 이득'], rows), '']
    # figure (horizontal bars so that the pair labels stay at 16 pt)
    fig, axes = plt.subplots(1, 2, figsize=(22, 13), constrained_layout=True, sharey=True)
    labs, vals = [], defaultdict(list)
    for r in g:
        if r['subset'] != 'all':
            continue
        labs.append(f"{r['setting']} {r['spanetv2'].split('spanetv2_')[1]} vs {CMP(r['comparison'])}")
        for k in ('gain_mIoU', 'gain_boundary_IoU', 'gain_interior_IoU'):
            vals[k].append(float(r[k]))
    y = np.arange(len(labs))[::-1]
    for i, (k, col, lab) in enumerate((('gain_mIoU', '#2a78d6', 'mIoU'), ('gain_boundary_IoU', '#eb6834', 'Boundary IoU'),
                                       ('gain_interior_IoU', '#1baf7a', 'interior IoU'))):
        axes[0].barh(y - (i - 1) * 0.27, vals[k], 0.25, color=col, label=lab, edgecolor='white', linewidth=1)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labs)
    axes[0].set_xlabel('gain (SPANetV2 − comparison), ADE20K val, all 2,000 images')
    axes[0].legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.06), ncol=3)
    axes[0].set_title('mIoU / Boundary IoU / interior IoU gain')
    for i, (sub_, col, lab) in enumerate((('thin_top10pct', '#2a78d6', 'thin-structure top 10 % (200 images)'),
                                          ('rest', '#898781', 'other 1,800 images'))):
        v = [float(r['gain_boundary_IoU']) for r in g if r['subset'] == sub_]
        axes[1].barh(y + (i - 0.5) * 0.4 * -1, v, 0.38, color=col, label=lab, edgecolor='white', linewidth=1)
    axes[1].set_xlabel('Boundary IoU gain (SPANetV2 − comparison)')
    axes[1].legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.06), ncol=2)
    axes[1].set_title('Boundary IoU gain by image subset')
    fig.savefig(os.path.join(FIGS, 'c_boundary_gains.png'), bbox_inches='tight')
    plt.close(fig)
    L += ['![boundary gains](figs/c_boundary_gains.png)', '']
    return L


# ================================================================================================ D
def sec_d():
    L = ['## D. (선택) SRF 추론 시 개입', '']
    rows = []
    for m in ('spanetv2_s18_pure', 'spanetv2_s18_hybrid'):
        for iv in ('none', 'flat'):
            d = jl(f'd_{m}_{iv}.json')
            if d:
                rows.append([NAME[m], '원래' if iv == 'none' else 'ψ 평탄화', d['n_srf_tensors_changed'], f"{d['clean_top1_val50k']:.2f}",
                             f"{d['clean_top1_1k']:.1f}", f"{d['fourier_mean_error']:.2f}", f"{d['fourier_mean_error_minus_clean']:.2f}",
                             f"{d['fourier_low_error']:.2f}", f"{d['fourier_rest_error']:.2f}", f"{d['pgd_high_eps1_10step_top1']:.1f}"])
    if not rows:
        return L + ['(실행하지 못함)', '']
    L += [table(['모델', 'SRF', '바뀐 SRF 텐서', 'clean top-1 (val 50k)', 'clean top-1 (1k)', 'Fourier 평균 오류', '− clean 오류',
                 'Fourier 저역(ρ≤0.25)', 'Fourier 나머지', 'high PGD ε=1, 10 step top-1'], rows), '']
    return L


def main():
    os.makedirs(FIGS, exist_ok=True)
    conc = {}
    p = os.path.join(HERE, 'conclusions.md')
    if os.path.exists(p):
        for block in re.split(r'^## ', open(p).read(), flags=re.M)[1:]:
            head, _, body = block.partition('\n')
            conc[head.strip()] = body.strip()
    La, _ = sec_a()
    secs = {'A': La, 'E': sec_e(), 'B': sec_b(), 'C': sec_c(), 'D': sec_d()}
    L = ['# 주파수 관련 결과의 해석 근거 (freq_interpretation, 추론만)', '', '{{INTRO}}', '']
    for k in ('A', 'E', 'B', 'C', 'D'):
        L += secs[k]
        if k in conc:
            L += [f'### 결론 ({k})', '', conc[k], '']
    if 'SUMMARY' in conc:
        L += ['## 요약: 네 문장에 대한 근거', '', conc['SUMMARY'], '']
    if 'RERUN' in conc:
        L += ['## 재실행과 파일', '', conc['RERUN'], '']
    intro = conc.get('INTRO', '')
    open(os.path.join(HERE, 'report.md'), 'w').write('\n'.join(L).replace('{{INTRO}}', intro))
    print('report.md written')


if __name__ == '__main__':
    main()
