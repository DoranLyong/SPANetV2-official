#!/usr/bin/env python
"""data/ -> report.md. The observation sentences are filled in from the data files (numbers only)."""
import json
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
AN = os.path.dirname(HERE)
REPO = os.path.dirname(AN)


def table(header, rows):
    esc = lambda c: str(c).replace('|', '\\|')
    return '\n'.join(['| ' + ' | '.join(esc(h) for h in header) + ' |', '|' + '|'.join('---' for _ in header) + '|'] +
                     ['| ' + ' | '.join(esc(c) for c in r) + ' |' for r in rows])


def main():
    hk = json.load(open(os.path.join(HERE, 'data', 'hooks.json')))
    imgs = json.load(open(os.path.join(HERE, 'data', 'images.json')))
    e1 = json.load(open(os.path.join(HERE, 'data', 'e1_stats.json')))
    e2 = json.load(open(os.path.join(HERE, 'data', 'e2_band_summary.json')))
    commit = subprocess.run(['git', '-C', REPO, 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    NAME = {'spanetv2_s18_pure': 'SPANetV2-S18-pure', 'convformer_s18': 'ConvFormer-S18', 'dfformer_s18': 'DFFormer-S18',
            'caformer_s18': 'CAFormer-S18'}
    L = ['# Token mixer의 주파수 처리 정성 비교 (추론만)', '',
         '학습과 미세조정은 하지 않았고 저장소 코드도 고치지 않았다(forward hook과 파라미터 읽기만 했다).',
         f'- **실행 환경:** SPANetV2-official commit `{commit}`, torch {hk["torch"]}, CPU, fp32, eval 모드.',
         '- **가중치:** ../freq_robustness와 같은 공개 가중치 파일이다(sha256 아래).',
         '- **다시 만들기:** `python select_images.py && python extract.py && python make_figs.py && python make_report.py` '
         '(앞의 둘은 spanetv2 conda env, 뒤의 둘은 spanetv2-dense env).', '',
         '## 모델, 가중치, hook 위치', '']
    rows = []
    for k, w in hk['weights'].items():
        rows.append([NAME[k], f"`{w['file']}`", f"`{w['sha256'][:16]}…`",
                     f"weights_only={w['weights_only']}, missing {len(w['missing'])}, unexpected {len(w['unexpected'])}"])
    L += [table(['모델', '가중치 파일', 'sha256', 'torch.load'], rows), '']
    rows = []
    for k, st in hk['hooks'].items():
        for s, info in st.items():
            rows.append([NAME[k], s, f"{info['block_index']} (0부터, 단계의 블록 {info['blocks_in_stage']}개 중 "
                                     f"{'마지막' if int(s) < 3 else '가운데'})", info['token_mixer'],
                         '<br>'.join(f'`{m}` → {d}' for m, d in info['hooks'].items())])
    L += [table(['모델', '단계', '블록 번호', 'token mixer', 'hook (모듈 이름 → 얻은 것)'], rows), '',
          '- **블록 위치:** 1·2단계는 마지막 블록, 3단계는 가운데 블록이다. 네 모델 모두 S18(블록 수 3-3-9-3)이라 같은 번호다.',
          '- **SPANetV2 head 출력:** `proj`의 입력 s_ctx는 ExSPAM.forward가 head마다 irfft2로 만든 s_i를 (B, C/4, 4, H, W)로 쌓아 '
          '(B, C, H, W)로 편 것이다. 그래서 view(B, C/4, 4, H, W)[:, :, i]가 head i(커널 3+2i)의 SRF 출력이다. 따로 다시 계산하지 않았다.',
          '- **DFFormer 필터 직후 출력:** `act2`는 nn.Identity이고, 그 입력이 동적 필터를 곱한 뒤 irfft2한 출력이다.',
          '- **DFFormer 예측 필터:** 예측 필터는 forward 안의 식 weight = einsum("bfc,hwf->bhwc", routing, complex_weights)와 같다. '
          '`reweight` 출력에서 얻은 routing(softmax)과 파라미터 basis `complex_weights`를 써서 만들었다.', '',
          '## 영상 12장', '',
          f'선택 규칙: {imgs["rule"]}', '']
    rows = [[i + 1, m['group'], m['source'], f"`{m['file']}`", f"{m['sobel_energy']:.4f}",
             '본문' if m['figure'] == 'main' else '부록'] for i, m in enumerate(imgs['images'])]
    L += [table(['#', '묶음', '출처', '파일', 'Sobel 에너지', '그림'], rows), '',
          f"- **Sobel 기준:** ImageNet val 전체의 Sobel 에너지 상위 25% 경계는 {imgs['sobel_top25_threshold']:.4f}이고, "
          f"전체 중앙값은 {imgs['sobel_energy_median_all']:.4f}이다.",
          '- **전처리:** 네 모델 모두 같은 평가 전처리(Resize 224 bicubic → CenterCrop 224, crop_pct 1.0)를 쓰므로 같은 crop을 넣었다.',
          '- **E2에 쓴 DFFormer 영상:** #1, #5, #9(세 묶음의 첫 영상)이다.', '',
          '## 정의', '',
          '- **스펙트럼:**',
          '  - 특징 X (C, H, W)에 대해 P = mean_c |FFT2(X_c)|²(전체 격자)를 구한다.',
          '  - L = log10(P / P_DC)를 정수 반경 고리 r = round(√(ky² + kx²)·H)로 평균한다.',
          '  - ρ = r / (H/2)로 둔다(축 방향 Nyquist = 1).',
          '- **고역/저역 power 비:** 선형 P를 ρ > 0.5 칸과 0 < ρ ≤ 0.25 칸에서 각각 평균한 뒤의 비로, DC는 뺐다.',
          '- **기울기:** 0 < ρ ≤ 1 고리에서 log10(P/P_DC)를 log10 ρ에 최소제곱으로 맞춘 기울기다.',
          '- **E1 지도:** 각 출력의 채널 L2 노름을 영상·출력마다 1·99 백분위로 0–1에 맞췄다.',
          '- **E2 정의(../srf_task_analysis, C-4와 같음):**',
          '  - 커널 응답 |K̂|는 그 단계 격자(H×H)에서의 DTFT로, zero-pad한 rfft2와 같다.',
          '  - |K̂|와 ψ·|K̂|는 채널마다 전 주파수(켤레 대칭 가중) 평균으로 나눈 뒤 채널 평균했다.',
          '  - ψ = sigmoid(M)은 나누지 않은 값이다.',
          '  - DFFormer 예측 필터 크기 |W|도 |K̂|와 같게 정규화했다.',
          '  - C-4 분석은 단계 안 모든 블록을 모았고, 여기는 위 표의 블록 하나만 쓴다.', '',
          '## 그림', '',
          '### E1. 실제 영상에서 mixer가 남기는 성분', '',
          '![e1 stage1](figs/e1_maps_stage1.png)', '', '![e1 stage2](figs/e1_maps_stage2.png)', '',
          '![e1 stage3](figs/e1_maps_stage3.png)', '',
          '- **`figs/e1_maps_stage{1,2,3}.png`:**',
          '  - 본문 영상 6장(ImageNet 무작위 3장, Sobel 상위 3장)에 대해, 블록의 token mixer 각 분기 출력에서 채널 L2 노름의 공간 분포를 그렸다.',
          '  - 열은 다음 순서다: 입력, ConvFormer depthwise conv 출력, DFFormer 동적 필터 직후 출력, SPANetV2 head k = 3, 5, 7, 9의 SRF 출력, SPANetV2 mixer 출력, (3단계) CAFormer attention 출력.',
          '- **부록 영상:** 나머지 6장은 `figs/e1_maps_stage{1,2,3}_appendix.png`에 있다.', '',
          '![e1 spectra](figs/e1_spectra.png)', '',
          '- **`figs/e1_spectra.png`:**',
          '  - 같은 출력들과 ConvFormer·DFFormer의 mixer 출력의 log10(P/P_DC) 반경 profile을 단계마다 겹쳤다.',
          '  - 선은 12장 평균이고 띠는 25–75%이다. 원자료는 `data/e1_radial.csv`에 있다.', '']
    rows = []
    for v in e1.values():
        rows.append([v['stage'], v['branch'], v['C'], f"{v['H']}×{v['H']}", f"{v['hi_lo_power_ratio_mean']:.4f}",
                     f"{v['hi_lo_power_ratio_p25']:.4f}–{v['hi_lo_power_ratio_p75']:.4f}", f"{v['slope_mean']:+.3f}",
                     f"{v['slope_p25']:+.3f} – {v['slope_p75']:+.3f}"])
    L += [table(['단계', '분기', '채널', '격자', '고역/저역 power 비 (12장 평균)', '25–75%', '기울기 (평균)', '25–75%'], rows), '']

    def r(s, br, key='hi_lo_power_ratio_mean'):
        return e1[f'{s}|{br}'][key]
    heads = ['SPANetV2|head k=3', 'SPANetV2|head k=5', 'SPANetV2|head k=7', 'SPANetV2|head k=9']
    obs1 = []
    for s in (1, 2, 3):
        hv = ', '.join(f'{r(s, h):.3f}' for h in heads)
        ex = f', CAFormer attention {r(3, "CAFormer|attention"):.3f}' if s == 3 else ''
        obs1.append(f'{s}단계 {hv} / ConvFormer {r(s, "ConvFormer|dwconv"):.3f} / DFFormer {r(s, "DFFormer|dynamic filter"):.3f}{ex}')
    sl = [(r(3, h, 'slope_mean')) for h in heads]
    L += ['**관찰 (E1).**',
          '- **고역/저역 power 비:** SPANetV2 head k = 3, 5, 7, 9 / ConvFormer depthwise / DFFormer 동적 필터 출력의 값은 다음과 같다: ' + '; '.join(obs1) + '.',
          f'- **3단계 반경 profile 기울기:** SPANetV2 head가 {max(sl):+.2f} ~ {min(sl):+.2f}이고, '
          f'ConvFormer {r(3, "ConvFormer|dwconv", "slope_mean"):+.2f}, DFFormer {r(3, "DFFormer|dynamic filter", "slope_mean"):+.2f}, '
          f'CAFormer {r(3, "CAFormer|attention", "slope_mean"):+.2f}이다.',
          f'- **mixer 출력의 같은 비:** 1단계에서 SPANetV2 {r(1, "SPANetV2|mixer"):.3f}, ConvFormer {r(1, "ConvFormer|mixer"):.3f}, '
          f'DFFormer {r(1, "DFFormer|mixer"):.3f}이고, 3단계에서 각각 {r(3, "SPANetV2|mixer"):.3f}, {r(3, "ConvFormer|mixer"):.3f}, '
          f'{r(3, "DFFormer|mixer"):.3f}이다.', '']

    L += ['### E2. 필터 응답', '',
          '![e2 stage1](figs/e2_filters_stage1.png)', '', '![e2 stage2](figs/e2_filters_stage2.png)', '',
          '![e2 stage3](figs/e2_filters_stage3.png)', '',
          '- **`figs/e2_filters_stage{1,2,3}.png`:** 윗줄은 fftshift한 2D 크기 응답(중심 = DC, 그림 안 공통 색 범위)이고, 아랫줄은 반경 profile(채널 평균과 25–75%)이다.',
          '  - ConvFormer: depthwise 7×7 커널의 |K̂|.',
          '  - SPANetV2: head k = 3, 5, 7, 9의 실효 응답 ψ·|K̂|.',
          '  - DFFormer: 영상 #1, #5, #9에 대해 그 블록이 예측한 필터 |W|. 원자료는 `data/e2_radial.csv`(|K̂|, ψ, ψ·|K̂| 따로)에 있다.',
          '- **DFFormer 필터의 구성:** DFFormer의 필터는 basis 4개(`complex_weights`)를 영상·채널마다 routing 가중치로 더한 것이다. '
          'basis와 12장의 routing은 `data/e2_dfformer_basis_and_routing.npz`에 저장했다.', '',
          '![e2 srf gain](figs/e2_srf_gain.png)', '',
          '- **`figs/e2_srf_gain.png`:** SPANetV2 head 4개의 SRF 이득 ψ = sigmoid(M)을 채널 평균해 단계마다 2D 지도로 그렸다.',
          '  - 색은 0.5(초기화 값)를 중심으로 한 발산형이다. 붉으면 초기값보다 더 통과시키고, 푸르면 덜 통과시킨다.', '']
    rows = []
    for v in e2.values():
        hl = v['band_high'] / v['band_low'] if v['band_low'] else float('nan')
        rows.append([v['stage'], v['branch'], f"{v['DC']:.3f}", f"{v['band_low']:.3f}", f"{v['band_mid']:.3f}",
                     f"{v['band_high']:.3f}", f'{hl:.3f}'])
    L += [table(['단계', '분기', 'DC', '저역 0<ρ≤0.25', '중역', '고역 ρ>0.5', '고역/저역'], rows), '',
          'ψ 행은 정규화하지 않은 이득이고, 나머지 행은 채널마다 전 주파수 평균으로 나눈 값이다.', '']

    def q(s, br):
        v = e2[f'{s}|{br}']
        return v['band_high'] / v['band_low']
    sp = {s: [q(s, f'SPANetV2 k={k} effective') for k in (3, 5, 7, 9)] for s in (1, 2, 3)}
    dfv = {s: [q(s, f'DFFormer predicted |filter| image {b}') for b in (1, 5, 9)] for s in (1, 2, 3)}
    psi_lo = [e2[f'{s}|SPANetV2 k={k} psi']['band_low'] for s in (1, 2, 3) for k in (3, 5, 7, 9)]
    psi_hi = [e2[f'{s}|SPANetV2 k={k} psi']['band_high'] for s in (1, 2, 3) for k in (3, 5, 7, 9)]
    L += ['**관찰 (E2).**',
          '- **SPANetV2 실효 응답과 ConvFormer |K̂|의 고역/저역 비:**',
          '  - SPANetV2 실효 응답 ψ·|K̂|는 단계마다 head 사이에서 '
          + ', '.join(f'{s}단계 {min(v):.2f}–{max(v):.2f}' for s, v in sp.items()) + '이다.',
          '  - ConvFormer |K̂|는 ' + ', '.join(f'{s}단계 {q(s, "ConvFormer |K| dwconv"):.2f}' for s in (1, 2, 3)) + '이다.',
          '- **DFFormer 예측 필터 |W|의 고역/저역 비:** 세 영상 사이에서 '
          + ', '.join(f'{s}단계 {min(v):.3f}–{max(v):.3f}' for s, v in dfv.items()) + '이다.',
          f'- **SPANetV2 ψ의 대역 평균:** 3단계 × head 4개에서 저역 {min(psi_lo):.3f}–{max(psi_lo):.3f}, '
          f'고역 {min(psi_hi):.3f}–{max(psi_hi):.3f}이다.', '',
          '## 파일', '',
          '- `select_images.py`: 영상 선택. 결과는 `data/images.json`, `data/images_u8.npy`이다.',
          '- `extract.py`: hook과 파라미터 읽기. 결과는 `data/hooks.json`, `data/e1_*`, `data/e2_params_*`이다.',
          '- `make_figs.py`: 그림, `data/e1_radial.csv`, `data/e2_radial.csv`, `data/e2_band_summary.json`, '
          '`data/e2_dfformer_basis_and_routing.npz`를 만든다.',
          '- `make_report.py`: 이 보고서를 만든다.',
          '- **CSV 열:** model, stage, branch, ring, rho, mean, p25, p75이다. E1의 p25·p75는 12장 사이 값이고, E2는 채널 사이 값이다.', '']
    open(os.path.join(HERE, 'report.md'), 'w').write('\n'.join(L))
    print('report.md written')


if __name__ == '__main__':
    main()
