#!/usr/bin/env python
"""data/ -> report.md (all numbers read from the data files; conclusion sentences are templated from them)."""
import csv
import json
import os
import subprocess
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(HERE, 'data')
CKPTS = ['IN1K', 'COCO/MaskRCNN', 'COCO/Cascade', 'ADE20K/FPN', 'ADE20K/UperNet']
SHOW = {'IN1K': 'IN1K', 'COCO/MaskRCNN': 'COCO MR', 'COCO/Cascade': 'COCO Cas', 'ADE20K/FPN': 'ADE FPN',
        'ADE20K/UperNet': 'ADE Uper'}
TR = ['conv', 'srf', 'exp', 'proj', 'mod_s', 'mod_q']
TRN = {'conv': 'conv x1→x2', 'srf': 'SRF x2→x3', 'exp': 'Exp 몫 x3→x4', 'proj': 'Σx4→s_ctx', 'mod_s': 's_ctx→m',
       'mod_q': 'q→m'}


def table(header, rows):
    esc = lambda c: str(c).replace('|', '\\|')
    return '\n'.join(['| ' + ' | '.join(esc(h) for h in header) + ' |', '|' + '|'.join('---' for _ in header) + '|'] +
                     ['| ' + ' | '.join(esc(c) for c in r) + ' |' for r in rows])


def rcsv(name):
    with open(os.path.join(DATA, name)) as f:
        return list(csv.DictReader(f))


def main():
    meta = json.load(open(os.path.join(DATA, 'images.json')))
    commit = subprocess.run(['git', '-C', REPO, 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    L = ['# SAG 분기별 활성 스펙트럼과 context 지도: ImageNet-1K, COCO, ADE20K 백본 비교 (추론만)', '',
         '학습, 미세조정, 저장소 코드 수정은 하지 않았다(forward hook만 썼다).',
         f'- **실행 환경:** SPANetV2-official commit `{commit}`, CPU, fp32, eval 모드.',
         '- **다시 만들기:** `bash run_all.sh`. 체크포인트 경로는 `extract.py --ckpt LABEL=PATH` 인자로 준다.',
         '- **범위:** 본문은 SPANetV2-S18-pure이다. S18-hybrid(단계 1–2만 SPAM)는 각 절 끝과 `_hybrid` 그림에 있다.', '',
         '## 1. 체크포인트와 불러오기', '']
    rows = []
    for arch in ('spanetv2_s18_pure', 'spanetv2_s18_hybrid'):
        for c in CKPTS:
            p = os.path.join(DATA, 'raw', arch, f'{c.replace("/", "_")}_load.json')
            if not os.path.exists(p):
                continue
            d = json.load(open(p))
            rows.append([arch.replace('spanetv2_', ''), c, f"`{os.path.basename(d['path'])}`", f"`{d['sha256'][:12]}…`",
                         f"weights_only={d['weights_only']}" + ('' if d['weights_only'] else ' (아래 주)'),
                         d['source'], d.get('n_non_backbone_keys_ignored', 0),
                         ', '.join(sorted({k.split('.')[0] for k in d['missing']})) or '없음',
                         ', '.join(d['unexpected']) or '없음'])
    L += [table(['모델', '체크포인트', '파일', 'sha256', 'torch.load', '쓴 state_dict', '버린 비백본 key 수',
                 '불러오지 않은 key (모듈 단위)', '남은 key'], rows), '',
          '- **weights_only:** 하류 체크포인트(mmengine 저장본)는 `weights_only=True`에서 `mmengine.logging.history_buffer.HistoryBuffer` '
          '때문에 실패한다. 저자 본인의 공개 파일이므로 `weights_only=False`로 읽었다.',
          '- **불러오지 않은 key:** 없다. 하류 백본(dense spanetv2.py)도 분류 head(`head.*`)와 마지막 `norm.*`을 가지고 있어 함께 채워졌다(stage 4 뒤라 측정에 쓰이지 않는다).',
          '- **남은 key:** 하류 백본의 출력 norm `norm0`–`norm3`은 분류 모델에 없어서 남았다.', '',
          '## 2. hook 위치와 Σx4_h 확인', '',
          'ExSPAM.forward(image_classification/models/spanetv2.py 237–269행)의 모든 SPAM 블록에 다음 hook을 걸었다.', '',
          table(['양', 'hook', '내용'], [
              ['x1_h', '`token_mixer.local_conv_{h+1}` forward hook 입력', 'ctx의 head h (B, C/4, H, W)'],
              ['x2_h', '같은 hook 출력', 'depthwise conv 출력'],
              ['x3_h', '`token_mixer.proj` forward-pre-hook 입력 s_in', 's_in.view(B, C/4, 4, H, W)[:, :, h], SRF 출력(irfft2 뒤)'],
              ['x4_h', '같은 pre-hook 안에서 계산', 'proj[0](head h 외 채널을 0으로 둔 s_in). Conv2d._conv_forward를 써서 hook이 다시 불리지 않게 함'],
              ['Σx4_h 확인', '`token_mixer.proj[0]` forward hook 출력', '실제 proj[0] 출력과 비교'],
              ['s_ctx', '`token_mixer.proj` forward hook 출력', 'proj 전체 출력(노트북의 vis_ctx, permute 전)'],
              ['q', '`token_mixer.query` forward hook 출력', 'query 출력 (B, H, W, C)'],
              ['m', '`token_mixer.proj_out` forward-pre-hook 입력', 'q ⊙ s_ctx']]), '']
    chk = []
    for arch in ('spanetv2_s18_pure', 'spanetv2_s18_hybrid'):
        for c in CKPTS:
            p = os.path.join(DATA, 'raw', arch, f'{c.replace("/", "_")}_load.json')
            if not os.path.exists(p):
                continue
            d = json.load(open(p))
            for R in (224, 512):
                v = d[f'R{R}']['sum_x4_check']
                chk.append([arch.replace('spanetv2_', ''), c, R, len(v), f"{max(x['max_abs_diff'] for x in v.values()):.2e}",
                            f"{max(x['rel_l2_max'] for x in v.values()):.2e}", 'none' if not any(x['bias'] for x in v.values()) else 'yes'])
    L += [table(['모델', '체크포인트', '해상도', 'SPAM 블록 수', 'max |Σx4_h − proj[0] 출력|', 'max ‖차‖/‖proj[0] 출력‖',
                 'proj[0] bias'], chk), '',
          '- **채널 배치:** s_in의 채널은 "채널 c의 head 0–3"이 연속으로 놓인다(c·4 + h).',
          '- **proj[0]:** groups = C/4인 1×1 conv라 그룹마다 이 4채널을 섞는다. 그래서 head별 Exp 출력은 없다.',
          '- **x4_h:** proj[0]이 선형이고 bias가 없으므로 x4_h의 합이 proj[0] 출력과 같아야 한다. 위 표의 차이는 부동소수 반올림 수준이다.', '',
          '## 3. 해상도, 전처리, 영상', '',
          '- **해상도:**',
          '  - R1 224×224에서는 SRF가 저장된 격자 그대로 쓰여 C-4와 비교된다.',
          '  - R2 512×512에서는 ExSPAM.forward가 SRF를 `resize_weight`(bicubic, align_corners=True)로 늘린다.',
          '  - 두 해상도 모두 분류 모델 그대로(input_size 224로 생성) 넣었다.',
          '- **전처리:** 짧은 변을 R로 PIL bicubic resize한 뒤 R×R center crop하고, (x/255 − ImageNet mean)/std로 정규화했다.',
          '- **하류 config와의 일치:** 하류 체크포인트 8개(S18-pure·hybrid)의 학습 config는 모두 data_preprocessor mean (123.675, 116.28, 103.53), '
          'std (58.395, 57.12, 57.375), bgr_to_rgb=True이다. [0,1] 척도의 ImageNet mean/std(RGB)와 같다.',
          '  - 하류 학습의 resize는 mmcv(cv2 bilinear)였고, 여기서는 PIL bicubic을 썼다.',
          '- **노트북(ExSPAM_analysis.ipynb 19–21셀)과 다른 점:**',
          '  - 노트북은 영상을 224×224로 늘린 뒤 Resize(224)를 했고, 여기서는 짧은 변 resize 뒤 center crop을 했다.',
          '  - 노트북은 지도를 ×4 bilinear로 올려 입력 없이 그렸고, 여기서는 지시대로 입력 크기로 bilinear로 올려 jet, alpha 0.5로 겹쳤다.', '',
          f"영상 선택 규칙: {meta['rule']}", '']
    rows = [[i + 1, m['group'], f"`{m['file']}`", ', '.join(m.get('imagenet_val_file') or []) or '',
             f"{m['object_area_R224']:.3f}" if m.get('object_area_R224') is not None else '—',
             '본문' if m['figure'] == 'main' else '부록'] for i, m in enumerate(meta['images'])]
    L += [table(['#', '묶음', '파일', 'ImageNet val 원본', '객체 면적 비 (R1)', '그림'], rows), '',
          '- **ImageNet 묶음:** `misc/imgs`의 10장 모두 ImageNet val 원본과 바이트가 같다(md5 확인). 지시대로 이 영상을 포함하되 하나도 빼지 않으려고 10장을 모두 썼다(다른 묶음은 8장).',
          '- **객체 면적이 0인 영상:** ADE20K 영상 가운데 thing 클래스 화소가 없는 영상은 객체 비율을 계산하지 않았다(면적 0).', '']

    # ---------------- spectra and gains
    bv = defaultdict(dict)
    for r in rcsv('band_values.csv'):
        bv[(r['arch'], r['res'], r['group'], r['stage'], r['quantity'], r['head'], r['band'])][r['ckpt']] = float(r['mean'])
    L += ['## 4. 스펙트럼과 전이 이득', '',
          '- **정의:**',
          '  - 텐서 X (C, H, W)마다 P = mean_c |FFT2(X_c)|²(ortho, 전체 격자)이다. profile은 log10(P/P_DC)를 정수 반경 고리로 평균한 것이고, ρ = r/(H/2)이다.',
          '  - 대역 값은 대역 안 격자 칸의 평균이다(low 0 < ρ ≤ 0.25, mid 0.25 < ρ ≤ 0.5, high ρ > 0.5, C-4와 같음).',
          '  - 전이 이득은 G = log10(P_out/P_in)이고, 블록마다 계산해 단계 안에서 평균했다.',
          '  - 통계는 영상 평균(전체 26장)이고, 25–75%는 `data/band_values.csv`에 있다.',
          '  - 블록별 원자료는 `data/raw/<arch>/<ckpt>_R<res>.npz`에, 고리별 값은 `data/profiles.csv.gz`, `data/transfer_gain.csv.gz`에 있다.', '']
    for s in (1, 2, 3, 4):
        L += [f'![transfer gain stage {s}](figs/transfer_gain_stage{s}.png)', '']
    L += ['- **`figs/transfer_gain_stage{1..4}.png`(R1):** 행은 전이 단계이고, 위 세 줄은 head(커널 3, 5, 7, 9)별 열로 나뉜다. 선은 체크포인트다.',
          '- **같은 형식의 다른 그림:** R2는 `_R512`, hybrid는 `_hybrid`.', '',
          '![sctx spectra](figs/sctx_spectra.png)', '',
          '- **`figs/sctx_spectra.png`:** s_ctx의 log10(P/P_DC) profile이다. 행은 해상도, 열은 단계다.', '']
    for R in ('224', '512'):
        rows = []
        for s in ('1', '2', '3', '4'):
            for bn in ('low', 'mid', 'high'):
                v = bv.get(('spanetv2_s18_pure', R, 'all', s, 's_ctx log10(P/P_DC)', '', bn))
                if v:
                    rows.append([s, bn] + [f'{v[c]:.3f}' for c in CKPTS])
        L += [f'**s_ctx 대역 값 log10(P/P_DC), S18-pure, {R}×{R}, 26장 평균**', '',
              table(['단계', '대역', *[SHOW[c] for c in CKPTS]], rows), '']
        for bn in ('low', 'high'):
            rows = []
            for s in ('1', '2', '3', '4'):
                for tr in TR:
                    for h in (['k=3', 'k=5', 'k=7', 'k=9'] if tr in ('conv', 'srf', 'exp') else ['']):
                        v = bv.get(('spanetv2_s18_pure', R, 'all', s, f'gain {tr}', h, bn))
                        if v:
                            rows.append([s, TRN[tr], h or '—'] + [f'{v[c]:+.3f}' for c in CKPTS])
            L += [f'**전이 이득 G, {bn} 대역, S18-pure, {R}×{R}, 26장 평균**', '',
                  table(['단계', '전이', 'head', *[SHOW[c] for c in CKPTS]], rows), '']

    # ---------------- contrast
    cs = rcsv('contrast_summary.csv')
    L += ['## 5. 과제 비교 (C-4 (e) 방식)', '',
          '- **비교 단위:** 칸 = (해상도, 영상 묶음, 단계, 양, head, 대역)이다. 양은 전이 이득 6종과 s_ctx 대역 값이다.',
          '- **W (과제 내):** 영상 평균값에서 |Mask R-CNN − Cascade|와 |FPN − UperNet| 가운데 큰 값이다.',
          '- **X (과제 간):** COCO 체크포인트와 ADE20K 체크포인트를 하나씩 짝지은 4쌍의 |차이|다.',
          '- **판정:** "X_min > W"는 4쌍 모두가 과제 내 차이보다 큰 칸이다. 칸별 값은 `data/contrast.csv`에 있다.', '']
    rows = []
    for r in cs:
        if r['quantity'] == 'all quantities' or r['group'] == 'all':
            n, k = int(r['cells']), int(r['cross_min_gt_within_max'])
            rows.append([r['arch'].replace('spanetv2_', ''), r['res'], r['group'], r['quantity'], n,
                         f'{k} ({100 * k / n:.0f}%)', r['median_Xmin_over_W']])
    L += [table(['모델', '해상도', '영상 묶음', '양', '칸 수', 'X_min > W', '중앙값 X_min/W'], rows), '']

    # ---------------- maps
    mc = rcsv('map_correlation_summary.csv')
    L += ['## 6. Context 지도', '',
          '![context overlay stage 4](figs/context_overlay_stage4.png)', '',
          '- **`figs/context_overlay_stage4.png`:** 본문 영상 9장(묶음마다 3장)의 4단계 context 지도다.',
          '  - 단계 안 모든 블록의 |s_ctx|를 채널 방향으로 이어 붙여 평균하고, 입력 크기로 bilinear 업샘플한 뒤 영상마다 1–99 백분위로 0–1에 맞췄다.',
          '  - jet 색, alpha 0.5로 입력 위에 겹쳤다. 열은 입력과 체크포인트 다섯이다.',
          '- **부록 그림:** 1–3단계, R2, 부록 영상, hybrid는 `figs/appendix_context_overlay_*.png`에 있다.', '',
          '지도 사이 Pearson 상관(같은 영상, 업샘플한 지도의 모든 화소, 영상 평균):', '']
    rows = []
    for arch in ('spanetv2_s18_pure', 'spanetv2_s18_hybrid'):
        for R in ('224', '512'):
            for s in ('1', '2', '3', '4'):
                d = {r['pair_type']: r for r in mc if r['arch'] == arch and r['res'] == R and r['stage'] == s and r['group'] == 'all'}
                if d:
                    rows.append([arch.replace('spanetv2_', ''), R, s] +
                                [f"{float(d[t]['mean_r']):.3f} [{float(d[t]['p25']):.3f}, {float(d[t]['p75']):.3f}]"
                                 for t in ('vs IN1K', 'within task', 'cross task')])
    L += [table(['모델', '해상도', '단계', 'IN1K 대비', '같은 과제 안', '과제 간'], rows), '',
          'pair별·묶음별 값은 `data/map_correlation.csv`와 `data/map_correlation_summary.csv`에 있다.', '']
    orows = rcsv('object_ratio.csv')
    agg = defaultdict(list)
    for r in orows:
        if r['energy_over_area'] != 'nan':
            agg[(r['arch'], r['res'], r['stage'], r['group'], r['ckpt'])].append(
                (float(r['context_energy_fraction_in_object']), float(r['object_area_fraction'])))
    rows = []
    for arch in ('spanetv2_s18_pure',):
        for R in ('224', '512'):
            for s in ('1', '2', '3', '4'):
                for g in ('coco', 'ade20k'):
                    cells = []
                    for c in CKPTS:
                        v = agg.get((arch, R, s, g, c))
                        if v:
                            e = np.mean([x[0] for x in v])
                            cells.append(f'{e:.3f}')
                    v = agg.get((arch, R, s, g, 'IN1K'))
                    if cells:
                        rows.append([R, s, g, len(v), f"{np.mean([x[1] for x in v]):.3f}"] + cells)
    L += ['객체 영역 안 context 에너지 비율(Σ_객체 지도 / Σ 지도, 영상 평균; 객체 면적 > 0인 영상만), S18-pure:', '',
          table(['해상도', '단계', '묶음', '영상 수', '객체 면적 비', *[SHOW[c] for c in CKPTS]], rows), '',
          '- **객체 영역:** COCO는 모든 인스턴스 mask의 합집합(crowd 포함)이다. ADE20K는 Mask2Former ADE20K-150 panoptic의 thing 클래스 100개 화소다.',
          '- **에너지 지도:** 정규화 전의 업샘플 지도(채널·블록 평균 |s_ctx|)다. 영상별 값과 hybrid는 `data/object_ratio.csv`에 있다.', '']

    # ---------------- conclusion
    allc = {(r['arch'], r['res'], r['group']): r for r in cs if r['quantity'] == 'all quantities'}

    def frac(a, R, g):
        r = allc[(a, R, g)]
        return int(r['cross_min_gt_within_max']), int(r['cells']), r['median_Xmin_over_W']
    pure = 'spanetv2_s18_pure'
    f224, f512 = frac(pure, '224', 'all'), frac(pure, '512', 'all')
    cor = {(r['res'], r['stage'], r['pair_type']): float(r['mean_r']) for r in mc if r['arch'] == pure and r['group'] == 'all'}
    L += ['## 7. 결론', '',
          f'- **과제 간 vs 과제 내:** S18-pure의 전이 이득과 s_ctx 대역 값에서 과제 간 4쌍의 차이가 모두 같은 과제 안의 차이보다 큰 칸은 '
          f'224×224에서 {f224[0]}/{f224[1]}칸({100 * f224[0] / f224[1]:.0f}%, 중앙값 X_min/W {f224[2]}), '
          f'512×512에서 {f512[0]}/{f512[1]}칸({100 * f512[0] / f512[1]:.0f}%, 중앙값 {f512[2]})이다(26장 전체).',
          f"- **4단계 context 지도의 Pearson 상관(224×224, 영상 평균):** IN1K 대비 {cor[('224', '4', 'vs IN1K')]:.3f}, "
          f"같은 과제 안 {cor[('224', '4', 'within task')]:.3f}, 과제 간 {cor[('224', '4', 'cross task')]:.3f}이다. "
          f"512×512에서는 각각 {cor[('512', '4', 'vs IN1K')]:.3f}, {cor[('512', '4', 'within task')]:.3f}, {cor[('512', '4', 'cross task')]:.3f}이다.",
          ('- **판정:** 두 해상도 모두 과제 간 차이가 과제 내 차이보다 큰 칸이 절반에 못 미쳐, 과제 간 차이는 대부분 같은 과제 안의 차이 수준이다.'
           if max(f224[0] / f224[1], f512[0] / f512[1]) < 0.5 else
           '- **판정:** 과제 간 차이가 과제 내 차이보다 큰 칸이 절반 이상인 해상도가 있다(위 비율).'), '',
          '## 8. 파일', '',
          '- `select_images.py`: 영상과 객체 mask. `extract.py`: hook과 스펙트럼(체크포인트 경로는 인자).',
          '- `analyze.py`: 단계 profile, 전이 이득, 과제 대비, 지도 상관과 객체 비율. `make_figs.py`, `make_report.py`, `run_all.sh`.',
          '- `data/`:',
          '  - `profiles.csv.gz`, `transfer_gain.csv.gz`: 고리별 값.',
          '  - `band_values.csv`: 대역 값.',
          '  - `contrast.csv`, `contrast_summary.csv`: 과제 대비.',
          '  - `map_correlation*.csv`, `object_ratio.csv`: 지도 상관과 객체 비율.',
          '  - `stage_maps_*.npz`: 업샘플한 지도.',
          '  - `raw/`: 블록별 원자료와 불러오기·Σx4 확인 json.',
          '  - `ade20k_150_isthing.json`: thing/stuff 목록과 출처.', '']
    open(os.path.join(HERE, 'report.md'), 'w').write('\n'.join(L))
    print('report.md written')


if __name__ == '__main__':
    main()
