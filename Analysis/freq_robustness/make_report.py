#!/usr/bin/env python
"""data/raw + data/*.json -> report.md (conclusions are read from conclusions.md, sections '## A', '## B', ...)."""
import glob
import json
import math
import os
import re
import subprocess

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, 'data', 'raw')
REPO = os.path.dirname(os.path.dirname(HERE))
CLS_ORDER = ['spanetv2_s18_pure', 'spanetv2_s18_hybrid', 'convformer_s18', 'caformer_s18', 'dfformer_s18',
             'cdfformer_s18', 'convnext_tiny', 'convnext_tiny_ema', 'swin_tiny', 'deit_small']
NAME = {'spanetv2_s18_pure': 'SPANetV2-S18-pure', 'spanetv2_s18_hybrid': 'SPANetV2-S18-hybrid',
        'convformer_s18': 'ConvFormer-S18', 'caformer_s18': 'CAFormer-S18', 'dfformer_s18': 'DFFormer-S18',
        'cdfformer_s18': 'CDFFormer-S18', 'convnext_tiny': 'ConvNeXt-T', 'convnext_tiny_ema': 'ConvNeXt-T (EMA 파일)',
        'swin_tiny': 'Swin-T', 'deit_small': 'DeiT-S'}
REF = {
    'spanetv2_s18_pure': (83.4, '학위논문 표 4.2'),
    'spanetv2_s18_hybrid': (83.9, 'SPANetV2-official README (표 4.2 값은 지시문에 없음)'),
    'convformer_s18': (83.0, 'sail-sg/metaformer README'),
    'caformer_s18': (83.6, 'sail-sg/metaformer README'),
    'dfformer_s18': (83.2, 'DFFormer 논문 Table 2 (README에 정확도 표 없음)'),
    'cdfformer_s18': (83.1, 'DFFormer 논문 Table 2 (README에 정확도 표 없음)'),
    'convnext_tiny': (None, 'README에 이 파일의 정확도 없음 (README IN-1K 표는 _ema 파일)'),
    'convnext_tiny_ema': (82.1, 'facebookresearch/ConvNeXt README'),
    'swin_tiny': (81.2, 'microsoft/Swin-Transformer README'),
    'deit_small': (79.9, 'facebookresearch/deit README_deit.md 표 (평가 출력 79.854)'),
}
CLS_FILE = {
    'spanetv2_s18_pure': 'SPANetV2-official image_classification/models/spanetv2.py',
    'spanetv2_s18_hybrid': 'SPANetV2-official image_classification/models/spanetv2.py',
    'convformer_s18': 'sail-sg/metaformer metaformer_baselines.py', 'caformer_s18': 'sail-sg/metaformer metaformer_baselines.py',
    'dfformer_s18': 'okojoalg/dfformer models/dfformer.py', 'cdfformer_s18': 'okojoalg/dfformer models/dfformer.py',
    'convnext_tiny': 'facebookresearch/ConvNeXt models/convnext.py', 'convnext_tiny_ema': 'facebookresearch/ConvNeXt models/convnext.py',
    'swin_tiny': 'microsoft/Swin-Transformer models/swin_transformer.py', 'deit_small': 'facebookresearch/deit models.py (timm VisionTransformer)',
}


def jl(name):
    p = os.path.join(RAW, name)
    return json.load(open(p)) if os.path.exists(p) else None


def f(x, nd=2):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return '—'
    return f'{x:.{nd}f}'


def table(header, rows):
    esc = lambda c: str(c).replace('|', '\\|')
    return '\n'.join(['| ' + ' | '.join(esc(h) for h in header) + ' |', '|' + '|'.join('---' for _ in header) + '|'] +
                     ['| ' + ' | '.join(esc(c) for c in r) + ' |' for r in rows])


def conclusions():
    p = os.path.join(HERE, 'conclusions.md')
    out = {}
    if os.path.exists(p):
        for block in re.split(r'^## ', open(p).read(), flags=re.M)[1:]:
            head, _, body = block.partition('\n')
            out[head.strip()] = body.strip()
    return out


def commits():
    out = {}
    for d in sorted(glob.glob(os.path.join(HERE, 'third_party', '*'))) + \
            [os.path.join(REPO, 'flops_benchmark', 'third_party', x) for x in ('metaformer', 'dfformer')]:
        h = subprocess.run(['git', '-C', d, 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
        if h:
            out[os.path.basename(d)] = h
    out['SPANetV2-official'] = subprocess.run(['git', '-C', REPO, 'rev-parse', 'HEAD'], capture_output=True,
                                              text=True).stdout.strip()
    return out


def sec0():
    L = ['## 0. 점검', '']
    rows = []
    man = json.load(open(os.path.join(HERE, 'data', 'weights_manifest.json')))
    for k in CLS_ORDER:
        d = jl(f'clean_{k}.json')
        if not d:
            continue
        ref, src = REF[k]
        li = d['load_info']
        p = d['preprocessing']
        rows.append([NAME[k], f(d['top1'], 2), f(ref, 1) if ref else '—', src,
                     f(d['top1'] - ref, 2) if ref else '—',
                     f"Resize {p['resize']} bicubic → CenterCrop 224 (crop_pct {p['crop_pct']})",
                     f"{li['state_dict']}, weights_only={li['weights_only']}, missing {len(li['missing'])}, unexpected {len(li['unexpected'])}",
                     f"`{os.path.basename(man[k]['url'])}`", f(d['fp16_top1'], 2)])
    L += [table(['모델', 'top-1 (fp32, val 50k)', '보고값', '보고값 출처', '차이', '전처리', '가중치 읽기', '가중치 파일',
                 '참고: fp16 autocast top-1'], rows), '']
    L += ['- **정규화:** 모든 모델이 mean (0.485, 0.456, 0.406), std (0.229, 0.224, 0.225)를 쓴다(각 저장소의 평가 코드에서 확인).',
          '- **보간:** Resize의 보간은 PIL bicubic이다.',
          '- **정밀도:** 0–C의 모든 수치는 fp32이다(TF32 끔).',
          '- **ConvFormer-S18·CAFormer-S18의 학위논문 값:** 학위논문 표 4.2의 값(각 83.0, 83.4)은 재학습한 결과다. '
          '이번 실험은 sail-sg/metaformer의 공개 가중치를 쓰므로 위의 보고값(README)과 대조했다. 공개 가중치 값과 재학습 값은 서로 다른 모델의 수치다.',
          '- **ConvNeXt-T:** 지시문의 `convnext_tiny_1k_224.pth`와 README IN-1K 표에 링크된 `convnext_tiny_1k_224_ema.pth`는 '
          '크기는 같고 sha256이 다른 별개 파일이다. A–C에는 지시문의 파일을 썼고, EMA 파일은 0에서만 재서 README 값(82.1)과 대조했다.',
          '- **SPANetV2-S18-hybrid의 기준값:** 지시문은 S18-pure의 학위논문 값(83.4)만 주었다. 그래서 hybrid는 저장소 README 값과 대조했다.', '']
    L += ['코드와 커밋은 다음과 같다(모델 파일은 경로로 불러왔고 수정하지 않았다).', '']
    c = commits()
    L += [table(['저장소', 'commit'], [[k, f'`{v}`'] for k, v in c.items()]), '']
    return L


def secA(C):
    L = ['## A. 대역 필터 검증 데이터의 정확도 곡선', '',
         '- **식:** Bai et al.(ECCV 2022) 식 (1)–(2)를 따른다. 채널마다 2D DFT를 하고 fftshift한 뒤 마스크를 곱한다. ifftshift와 역 DFT를 거쳐 실수부를 쓴다. clamp는 하지 않는다.',
         '- **마스크 모양:** HAT 공식 저장소(jiawangbai/HAT)에는 필터 평가 코드가 공개되어 있지 않다(커밋 이력, 이슈 포함해 확인).',
         '  - 인쇄된 식은 거리로 min(|i − H/2|, |j − W/2|)를 쓴다. 이대로면 십자 모양이 된다.',
         '  - 그래서 지시대로 정사각형(Chebyshev, max)으로 구현했다.',
         '  - 저역 M_l^S는 max(|i − 112|, |j − 112|) ≤ S/2, 고역 M_h^S는 같은 거리 > (224 − S)/2이다. 부등호는 식대로 비엄격(≤)이다.',
         '- **필터 크기:** 저역 S = 20, 40, …, 220, 고역 S = 170, 180, …, 220.',
         '- **평가 영상:** ImageNet-1K val 50,000장 전체, 각 모델의 평가 전처리 뒤 [0,1] 화소에 필터를 적용했다.',
         '- **원문과 다른 점:** 원문 그림 2의 정확한 S 표본(축 눈금 20/60/…/220, 170–220)과 필터를 정규화 전에 적용했는지는 원문에 적혀 있지 않다.',
         '', '![band filter curves](figs/band_filter_curves.png)', '', '![band filter examples](figs/band_filter_examples.png)', '']
    rows = []
    for k in CLS_ORDER:
        d = jl(f'band_{k}.json')
        if not d:
            continue
        t = d['top1']
        rows.append([NAME[k]] + [f(t[f'low{S}']) for S in range(20, 221, 20)] + [f(t[f'high{S}']) for S in range(170, 221, 10)])
    L += [table(['모델'] + [f'L{S}' for S in range(20, 221, 20)] + [f'H{S}' for S in range(170, 221, 10)], rows), '',
          'L = 저역 통과 크기 S, H = 고역 통과 크기 S, top-1 (%).', '']
    if 'A' in C:
        L += ['**결론 (A).** ' + C['A'], '']
    return L


def secB(C):
    L = ['## B. Fourier heat map', '',
         '- **방법:** Yin et al.(NeurIPS 2019)을 Bai et al. 5.6절의 설정으로 따랐다.',
         '  - 주파수 (u, v)마다 단일 Fourier 기저 U_{u,v}를 만든다. ‖U‖₂ = 1이고, fftshift 스펙트럼의 (u, v)에 0.5+0.5j, 대칭 좌표에 0.5−0.5j를 둔다.',
         '  - 영상 x ∈ [0,1]에 x + r·v·U를 더한다. v = 15.7은 채널마다의 ℓ2 노름이고, 세 채널에 같은 U를 더한다.',
         '  - 부호 r ∈ {−1, 1}은 영상·채널마다 독립으로 뽑았다(원문 "perturb every channel independently"). clamp는 하지 않았다.',
         '- **척도 확인:** Yin et al. 원문("pixels take values in [0,1]", 각 채널 독립)과 공식 코드(google-research frequency_analysis/freq_heatmap.py)를 확인했다.',
         '  - 공식 코드는 같은 기저를 모든 채널에 더한다. 그래서 v는 채널당 노름이고 전체 노름은 v·√3이다.',
         '  - 공식 코드의 기본값(rand_flip=False)은 부호를 바꾸지 않는다. 여기서는 원문 기술(무작위 부호)을 따랐다.',
         '- **원문 대비 표본:** 표본은 클래스마다 5장, 5,000장(numpy default_rng(0))이다. Bai et al.과 Yin et al.은 val 50,000장 전체를 썼다.',
         '- **표시 영역과 대역 평균:**',
         '  - 중심 63×63 영역의 오프셋 |u|, |v| ≤ 31은 ρ = |freq|/112 ≤ 0.28(모서리 0.39)에 해당한다. 그래서 지시문의 고역(ρ > 0.5)이 이 영역 안에 없다.',
         '  - 고역 평균과 반경 곡선을 위해, 63×63 영역 밖 전 스펙트럼을 stride 8 격자(켤레 쌍 369개, 그중 ρ > 0.5인 것 319개)로 추가로 쟀다.',
         '  - 저역 평균(ρ ≤ 0.25)은 63×63 영역에서, 고역 평균(ρ > 0.5)은 stride 8 격자에서 계산했다.',
         '', '![fourier heat maps](figs/fourier_heatmaps.png)', '', '![fourier radial](figs/fourier_radial.png)', '']
    rows = []
    for k in CLS_ORDER:
        if not os.path.exists(os.path.join(RAW, f'fourier_{k}.npy')):
            continue
        g = np.load(os.path.join(RAW, f'fourier_{k}.npy')) * 100
        H = g.shape[0] // 2
        u, v = np.meshgrid(np.arange(-H, H + 1), np.arange(-H, H + 1), indexing='ij')
        r = np.sqrt(u ** 2 + v ** 2) / 112
        sp = jl(f'fourier_sparse_{k}.json')
        hi = md = None
        if sp:
            pr = np.array([(np.hypot(*map(int, key.split(','))) / 112, e * 100) for key, e in sp['error'].items()])
            hi = pr[pr[:, 0] > 0.5, 1].mean()
            md = pr[(pr[:, 0] > 0.25) & (pr[:, 0] <= 0.5), 1].mean()
        rows.append([NAME[k], f(np.nanmean(g[r <= 0.25 + 1e-9])), f(np.nanmean(g[(r > 0.25) & (r <= 0.5)])),
                     f(md), f(hi), f(np.nanmean(g)), f(np.nanmax(g))])
    L += [table(['모델', '저역 ρ ≤ 0.25 (63×63)', '0.25 < ρ ≤ 0.39 (63×63 안)', '중역 0.25 < ρ ≤ 0.5 (stride 8)',
                 '고역 ρ > 0.5 (stride 8)', '63×63 전체 평균', '63×63 최대'], rows), '', '값은 오류율(%)이다.', '']
    if 'B' in C:
        L += ['**결론 (B).** ' + C['B'], '']
    return L


def secC(C):
    L = ['## C. 대역을 제한한 적대적 공격', '',
         '- **공격 설정:**',
         '  - L2 PGD, ε ∈ {0.25, 0.5, 1.0}([0,1] 화소, 224×224×3), 10 step, step 크기 ε/4.',
         '  - 무작위 시작 없음, untargeted, cross-entropy, fp32. 표본 5,000장(B와 같은 집합).',
         '- **매 step의 순서:**',
         '  1. δ ← δ + α·g/‖g‖₂.',
         '  2. 대역 사영 δ ← irfft2(rfft2(δ, norm="ortho") ⊙ M, norm="ortho"). M은 켤레 대칭인 실수 마스크로, 저역은 M_l^56(A와 같은 정사각형, max ≤ 28), 고역은 그 여집합이다.',
         '  3. L2 공 δ ← δ·min(1, ε/‖δ‖₂).',
         '  4. δ ← clamp(x + δ, 0, 1) − x.',
         '- **clamp가 대역을 깨뜨리는 정도:** 마지막 clamp 때문에 δ의 일부가 대역 밖으로 나간다. 마지막 step 뒤 δ 에너지 가운데 대역 밖 비율의 평균을 아래 표에 함께 적었다.',
         '', '![pgd band](figs/pgd_band.png)', '']
    rows = []
    for k in CLS_ORDER:
        d = jl(f'pgd_{k}.json')
        if not d:
            continue
        r = d['results']
        cells = [NAME[k], f(d['clean_top1'])]
        for v in ('full', 'low', 'high'):
            for e in ('0.25', '0.5', '1.0'):
                cells.append(f(r[f'{v}|{e}']['top1']))
        cells.append(', '.join(f(r[f'{v}|1.0']['out_of_band_energy_mean'] * 100, 2) for v in ('low', 'high')))
        rows.append(cells)
    L += [table(['모델', 'clean'] + [f'{v} ε={e}' for v in ('제한없음', '저역', '고역') for e in ('0.25', '0.5', '1.0')] +
                ['대역 밖 에너지 % (저역, 고역; ε=1.0)'], rows), '', 'top-1 (%), 5,000장.', '']
    if 'C' in C:
        L += ['**결론 (C).** ' + C['C'], '']
    return L


SPANETV2_SEG_REF = {  # SPANetV2-official README (single scale)
    'upernet_spanetv2_s18_pure': 48.7, 'upernet_spanetv2_s18_hybrid': 49.1, 'upernet_spanetv2_s36_pure': 49.8,
    'upernet_spanetv2_s36_hybrid': 51.6, 'fpn_spanetv2_s18_pure': 46.7, 'fpn_spanetv2_s18_hybrid': 47.8,
    'fpn_spanetv2_s36_pure': 47.9, 'fpn_spanetv2_s36_hybrid': 48.6}


def secD(C):
    L = ['## D. 미세 경계 지표', '',
         '- **통제 비교가 아님:** 같은 골격의 비교 모델(ConvFormer, CAFormer, DFFormer, CDFFormer)은 하류 과제 가중치가 공개되어 있지 않다.',
         '  - 그래서 골격이 다르고 공식 가중치가 공개된 백본과 비교했다.',
         '  - 사전학습, 학습 일정, 학습 설정, 디코더 채널이 모델마다 달라 통제 비교가 아니다.',
         '- **실행 환경:** 모두 한 환경(SPANetV2-official의 mmseg 1.x / mmdet 3.x 사본, mmcv 2.1.0, torch 2.1.0)에서 단일 스케일, flip 없이 추론했다.',
         '  - SPANetV2는 체크포인트에 저장된 학습 config를 그대로 썼다.',
         '  - 비교 모델(mmcv 1.x 저장소)은 공식 백본 파일을 수정 없이 경로로 불러왔다. 불러오는 동안만 `mmcv_custom`, `*.models.builder`, `get_root_logger`를 대신하는 작은 객체를 sys.modules에 두었다(seg_eval.py, det_eval.py 주석).',
         '  - head와 데이터 설정은 공식 config를 mmseg 1.x / mmdet 3.x 형식으로 옮겼다.',
         '  - 가중치 key는 변환 없이 그대로 들어갔다. 아래 표의 missing / unexpected 열 참고.',
         '- **PoolFormer:** sail-sg/poolformer README의 FPN 가중치는 Google Drive에만 있는데, 이 서버에서는 접속이 되지 않는다(SSL_ERROR_SYSCALL). 그래서 지시문이 허용한 mmseg model zoo의 같은 모델 가중치와 config를 썼다.',
         '- **Boundary IoU:**',
         '  - Cheng et al.(CVPR 2021)의 정의와 기본 경계 폭 d = 영상 대각선의 2%를 따랐고, 공식 boundary-iou-api의 `mask_to_boundary`를 그대로 썼다.',
         '  - API에 의미론적 분할 모드가 없어서 클래스마다 GT·예측 이진 마스크의 경계 영역 교집합과 합집합을 val 전체에서 합했다.',
         '  - 클래스별 IoU를 낸 뒤 평균했다(mIoU와 같은 방식, GT가 255인 화소 제외).',
         '- **PoolFormer 테스트 전처리:** 체크포인트 안의 학습 config(mmseg 0.24)는 `AlignResize(keep_ratio, size_divisor=32)`이다. 여기서는 mmseg 1.x model zoo config의 Resize + ResizeToMultiple(32)을 썼다.',
         '- **COCO Boundary AP:** 같은 API의 COCO 평가(iouType="boundary", dilation 0.02, mask IoU와 boundary IoU의 min)를 썼다.',
         '  - API(2021)는 numpy 1.24에서 없어진 `np.float`를 쓴다. 그래서 API 코드는 고치지 않고 평가 프로세스 안에서만 `np.float = float`를 두었다.',
         '  - 같은 API의 segm 평가로 잰 mask AP를 mmdet CocoMetric의 mask AP와 대조했다.',
         '- **다중 스케일 값과의 관계:** 학위논문 표 4.6의 Swin-T·ConvNeXt-T UperNet 값은 다중 스케일이다. 여기 값은 모두 단일 스케일이고, 공개 수치도 단일 스케일 값과 대조했다.', '']
    for setting, title in (('upernet', 'D1. ADE20K UperNet (160k, 512×512)'), ('fpn', 'D2. ADE20K Semantic FPN (40k, 512×512)')):
        rows = []
        for p in sorted(glob.glob(os.path.join(RAW, 'seg_*.json'))):
            d = json.load(open(p))
            if d['setting'] != setting:
                continue
            b = jl(f'boundary_{d["model"]}.json')
            ref = d.get('reference') or {}
            li = d['load_info']
            rows.append([d['pair'], d['name'], f(d['mIoU']), f(b['mIoU']) if b else '—', f(b['boundary_IoU']) if b else '—',
                         f(ref.get('mIoU_ss')) if ref else f(SPANETV2_SEG_REF.get(d['model'])), d['test_cfg'].get('mode', ''),
                         f"weights_only={li['weights_only']}, missing {len(li['missing'])}, unexpected {len(li['unexpected'])}"])
        rows.sort(key=lambda r: (r[0], r[1]))
        L += [f'### {title}', '', table(['짝', '모델', 'mIoU (이번)', 'mIoU (저장 예측에서 재계산)', 'Boundary IoU',
                                          '공개 mIoU (단일 스케일)', 'test mode', '가중치 읽기'], rows), '',
              '공개 mIoU의 출처: SPANetV2는 SPANetV2-official README, ConvNeXt는 facebookresearch/ConvNeXt semantic_segmentation '
              'README의 ss 값, Swin은 Swin-Transformer-Semantic-Segmentation README의 단일 스케일 값, PoolFormer는 mmseg model zoo '
              'README의 mIoU*(Resize + ResizeToMultiple; ResizeToMultiple 없이 S24 40.12, S36 41.61; 저장소 README 40.3, 42.0)이다.', '']
        if f'D{1 if setting == "upernet" else 2}' in C:
            L += [f'**결론 (D{1 if setting == "upernet" else 2}).** ' + C[f'D{1 if setting == "upernet" else 2}'], '']
    rows = []
    for p in sorted(glob.glob(os.path.join(RAW, 'det_*.json'))):
        d = json.load(open(p))
        b = jl(f'boundary_ap_{d["model"]}.json')
        li = d['load_info']
        rows.append([d['framework'], d['pair'], d['name'], f(d['metrics'].get('bbox_mAP', float('nan')) * 100, 1),
                     f(d['metrics'].get('segm_mAP', float('nan')) * 100, 1), f(b['boundary']['AP'], 1) if b else '—',
                     f(d['reference']['box_AP'], 1), f(d['reference']['mask_AP'], 1),
                     f"weights_only={li['weights_only']}, missing {len(li['missing'])}, unexpected {len(li['unexpected'])}"])
    rows.sort(key=lambda r: (r[0], r[1], r[2]))
    L += ['### D3. COCO val2017 (3× 일정, 단일 스케일 1333×800)', '',
          table(['프레임워크', '짝', '모델', 'box AP', 'mask AP', 'Boundary AP', '공개 box AP', '공개 mask AP', '가중치 읽기'], rows), '']
    if 'D3' in C:
        L += ['**결론 (D3).** ' + C['D3'], '']
    L += ['![boundary examples](figs/boundary_examples.png)', '']
    ex = json.load(open(os.path.join(HERE, 'data', 'boundary_examples.json'))) if os.path.exists(
        os.path.join(HERE, 'data', 'boundary_examples.json')) else None
    if ex:
        L += [f"경계 오류 지도 영상 선택 규칙: {ex['rule']}. 고른 영상: {', '.join(ex['images'])}.", '']
    return L


def main():
    C = conclusions()
    L = ['# 주파수 대역별 강건성과 미세 경계 평가 (추론만)', '',
         '학습과 미세조정은 하지 않았고 저장소 코드도 고치지 않았다. 다시 실행하는 방법은 끝의 "파일" 절에 있다.', '']
    L += sec0() + secA(C) + secB(C) + secC(C) + secD(C)
    L += ['## 파일', '',
          '- `download_weights.py`: 가중치를 파일마다 새 디렉터리(`weights/<group>/<이름>/`)로 받는다. `data/weights_manifest.json`에 URL과 sha256을 적는다.',
          '- `make_cache.py`: ImageNet val을 모델별 평가 전처리(crop_pct 1.0 / 0.875)로 한 번 디코딩해 uint8로 저장한다. 5,000장 표본도 만든다.',
          '- `exp_cls.py`: 0, A, B, C 실행(`--weight`로 가중치 경로를 받는다). `scheduler.py`는 GPU마다 작업을 배분한다.',
          '- `seg_eval.py`, `det_eval.py`: D 추론. `boundary_eval.py`: mIoU, Boundary IoU, 예시 영상 선택. `boundary_ap.py`: COCO Boundary AP.',
          '- `make_figs.py`: figs/와 `data/summary.csv`(열 model, experiment, setting, value).',
          '- `write_conclusions.py`: 실험별 결론 문장(`conclusions.md`, 수치는 data/raw에서 채움). `make_report.py`: 이 보고서.',
          '- `data/raw/`: 실험별 원자료(json, npy).', '']
    with open(os.path.join(HERE, 'report.md'), 'w') as fh:
        fh.write('\n'.join(L))
    print('report.md written')


if __name__ == '__main__':
    main()
