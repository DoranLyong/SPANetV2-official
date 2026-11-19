"""
Write README.md from results/results.json, results/env.json and results/config_check.txt.

    python report.py
"""
import json
import os
import re
from decimal import ROUND_HALF_UP, Decimal

import zoo

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, 'results', 'results.json')))
ENV = json.load(open(os.path.join(HERE, 'results', 'env.json')))
CFG = open(os.path.join(HERE, 'results', 'config_check.txt')).read()

ORDER = [m[0] for m in zoo.MODELS if m[0] in R]
MAIN = [n for n in ORDER if R[n]['group'] != 'GFNet']
GFNET = [n for n in ORDER if R[n]['group'] == 'GFNet']
SPEC_COMMITS = {'spanet': '886ad24', 'spanetv2': '1b2cfee', 'metaformer': '64c534a', 'dfformer': '2b757c5', 'gfnet': '-'}
CAT_LABEL = {'a': '(a) FFT', 'b': '(b) 주파수 영역 곱', 'c': '(c) 그 밖의 원소별',
             'zero': '0 (자료 이동)', 'ignored': '0 (fvcore 무시 목록, 참고)'}
NORM_OPS = ('layer_norm', 'group_norm', 'batch_norm', 'instance_norm')


def G(x):
    return '-' if x is None else f'{x / 1e9:.3f}'


def M(x):
    return f'{x / 1e6:.3f}'


def r1(x):
    """Round a MACs count to 0.1 G, half up."""
    return float(Decimal(str(x / 1e9)).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP))


def r0(x):
    return int(Decimal(str(x / 1e6)).quantize(Decimal('1'), rounding=ROUND_HALF_UP))


def paper(r):
    return f"{r['paper_params_M']} / {r['paper_macs_G']}"


def table(header, names, row_fn, groups=True):
    lines = ['| ' + ' | '.join(header) + ' |', '|' + '|'.join(['---'] * len(header)) + '|']
    cur = None
    for n in names:
        if groups and R[n]['group'] != cur:
            cur = R[n]['group']
            lines.append('| **' + cur + '** ' + '| ' * (len(header) - 1) + '|')
        lines.append('| ' + ' | '.join(str(c) for c in row_fn(R[n])) + ' |')
    return '\n'.join(lines)


def change(reported, measured, unit, q):
    """'논문 → 재측정 (차이)': the measured value is rounded half up to the paper's precision `q`
    ('1' for Params in M, '0.1' for MACs in G); the difference is taken between the rounded values."""
    v = Decimal(str(measured / unit)).quantize(Decimal(q), rounding=ROUND_HALF_UP)
    p = Decimal(str(reported)).quantize(Decimal(q))
    d = v - p
    return f"{p} → {v} ({'±0' if d == 0 else f'{d:+}'})"


def main_row(r):
    ga, a, b, c = r['ga'], r['a'], r['b'], r['c']
    return [r['name'], change(r['paper_params_M'], r['params'], 1e6, '1'),
            change(r['paper_macs_G'], r['total'], 1e9, '0.1'),
            G(ga), G(r['na']), G(ga + a), G(ga + a + b), G(ga + a + b + c), G(r['total'])]


def increments_row(r):
    relu = sum(x['elements'] for x in r['convention_records'] if x['cat'] == 'ignored')
    return [r['name'], f"{r['a']:,}", f"{r['b']:,}", f"{r['c']:,}", f"{r['a'] + r['b'] + r['c']:,}", f'{relu:,}']


def notebook_row(r):
    p = r['paper_macs_G']
    ga1, na1 = r1(r['ga']), r1(r['na'])
    match = [k for k, v in (('(가)', ga1), ('(나)', na1)) if v == p]
    return [r['name'], p, G(r['ga']), G(r['na']), ga1, na1, r['na_calls'], ', '.join(match) if match else '둘 다 다름']


def fcm_row(r):
    norms = sum(v for k, v in r['ga_by_operator'].items() if k in NORM_OPS)
    fcm_ops = ', '.join(f'{k.replace("aten.", "")} {v / 1e9:.3f}' for k, v in r['fcm_by_op'].items())
    return [r['name'], G(r['ga']), G(r['fcm_flops']), G(r['fcm_flops'] / 2), G(norms),
            f"{round(r['ga'] - norms - r['fcm_flops'] / 2):,}", fcm_ops]


def fft_check_rows():
    lines = ['| 모델 | 연산 | 변환 | 모듈 | 입력 shape (dtype) | 출력 shape (dtype) | dim | s | E 텐서 | E | N | ⌈log2 N⌉ | E·⌈log2 N⌉ |',
             '|' + '---|' * 13]
    for n in ORDER:
        for f in R[n]['fft_checks']:
            lines.append(f"| {n} | `{f['op']}` | {'실수' if f['transform'] == 'real' else '복소'} | `{f['module']}` | "
                         f"{tuple(f['in_shape'])} ({f['in_dtype']}) | {tuple(f['out_shape'])} ({f['out_dtype']}) | "
                         f"{f['dim']} | {f['s']} | {'입력' if f['E_tensor'] == 'input' else '출력'} | {f['E']:,} | "
                         f"{f['N']:,} | {f['ceil_log2_N']} | {f['count']:,} |")
    return '\n'.join(lines)


def nb_check_rows():
    lines = ['| 모델 | 연산 | 모듈 | inputs[0] shape (dtype) | B, H, W, C 로 푼 값 | N = H·W | 노트북 함수 값 | 새 규약 값 (같은 호출) |',
             '|' + '---|' * 8]
    for n in ORDER:
        r = R[n]
        for c in r.get('na_check') or []:
            u = c['unpacked']
            ours = next((f['count'] for f in r['fft_checks'] if f['op'] == c['op']), None)
            lines.append(f"| {n} | `{c['op']}` | `{c['module']}` | {tuple(c['in_shape'])} ({c['in_dtype']}) | "
                         f"{u['B']}, {u['H']}, {u['W']}, {u['C']} | {u['H'] * u['W']:,} | {int(c['value']):,} | "
                         f"{ours:,} |")
    return '\n'.join(lines)


def unsupported_block(r):
    rows = r['unsupported_by_module']
    lines = [f"`fvcore.nn.FlopCountAnalysis(model, x).unsupported_ops()` 출력 원문:", '', '```', r['unsupported_ops_raw'], '```', '',
             '| 연산 | 호출 횟수 (전체) | 모듈 (이름 패턴 [클래스] 호출 횟수) |', '|---|---|---|']
    for op, total in sorted(r['unsupported_ops'].items(), key=lambda kv: (-kv[1], kv[0])):
        where = '; '.join(f"`{x['module']}` [{x['cls']}] {x['calls']}" for x in rows if x['op'] == op)
        lines.append(f'| `{op}` | {total} | {where} |')
    return '\n'.join(lines)


def convention_block(r):
    lines = ['| 항목 | 연산 | 출력 dtype | 모듈 (이름 패턴) | 호출 | 원소 수 (FFT는 ΣE) | 셈 값 (MACs) | reduction 입력 원소 수 |',
             '|---|---|---|---|---|---|---|---|']
    order = {'a': 0, 'b': 1, 'c': 2, 'zero': 3, 'ignored': 4}
    for x in sorted(r['convention_records'], key=lambda x: (order[x['cat']], x['op'], x['module'])):
        red = f"{x['input_numel']:,}" if 'input_numel' in x else ''
        lines.append(f"| {CAT_LABEL[x['cat']]} | `{x['op']}` | {x['dtype']} | `{x['module']}` | {x['calls']} | "
                     f"{x['elements']:,} | {x['count']:,} | {red} |")
    lines.append(f"\n합계: (a) {r['a']:,} / (b) {r['b']:,} / (c) {r['c']:,} MACs")
    return '\n'.join(lines)


def cfg_block(names):
    blocks = re.split(r'\n(?=== )', CFG.strip())
    keep = [b for b in blocks if any(b.startswith(f'== {n}:') for n in names)]
    return '```\n' + '\n\n'.join(keep) + '\n```'


def env_section():
    e = ENV
    lines = ['| 항목 | 값 |', '|---|---|']
    for k in ('date', 'python', 'torch', 'fvcore', 'timm', 'numpy', 'device', 'torch_threads', 'cpu', 'platform',
              'python_executable'):
        lines.append(f'| {k} | `{e[k]}` |')
    lines += ['', '| 저장소 | 측정 파일 | 측정 시 커밋 | 지정 커밋 | 측정 파일 상태 | remote |', '|---|---|---|---|---|---|']
    for key, v in e['repos'].items():
        spec = SPEC_COMMITS[key]
        ok = '' if spec == '-' else (' (일치)' if v['commit'].startswith(spec) else ' (다름)')
        lines.append(f"| {key} | `{v['file']}` | `{v['commit']}` | `{spec}`{ok} | {v['file_status']} | {v['remote']} |")
    return '\n'.join(lines)


def diff_lines():
    ga_diff = sorted(((R[n]['ga'] / 1e9 - R[n]['paper_macs_G'], n) for n in ORDER), key=lambda t: -abs(t[0]))
    na_over = [(R[n]['na'] / 1e9 - R[n]['paper_macs_G'], n) for n in ORDER if R[n]['na'] is not None
               and abs(R[n]['na'] / 1e9 - R[n]['paper_macs_G']) > 0.1]
    p_diff = [(n, R[n]['params'], R[n]['paper_params_M']) for n in ORDER if r0(R[n]['params']) != R[n]['paper_params_M']]
    return ga_diff, na_over, p_diff


def main():
    ga_diff, na_over, p_diff = diff_lines()
    spanetv2 = [n for n in ORDER if R[n]['repo'] == 'spanetv2']
    spanet = [n for n in ORDER if R[n]['repo'] == 'spanet']
    ex = R['SPANetV2-S18-pure']
    ex_nb = {c['op']: c for c in ex['na_check']}
    ex_ft = {f['op']: f for f in ex['fft_checks']}
    rf, nb_rf, nb_ir = ex_ft['aten::fft_rfft2'], ex_nb['aten::fft_rfft2'], ex_nb['aten::fft_irfft2']
    ops_all = sorted({op for n in ORDER for op in R[n]['unsupported_ops']})
    spanet_trainable = {n: R[n]['params'] - R[n]['params_trainable'] for n in spanet}

    md = []
    w = md.append
    w('# 연산량 재측정: SPANet·SPANetV2와 MetaFormer·DFFormer 비교 모델\n')
    w('학위논문 표의 연산량은 기준이 섞여 있다. SPANet·SPANetV2는 각 저장소의 `FLOPs_count.ipynb`(fvcore)로 쟀고, '
      '비교 모델은 원논문 보고값을 옮겼다. 이 폴더에서는 모든 모델을 **같은 환경, 같은 스크립트, 같은 규약**으로 다시 쟀다. '
      '모델 코드와 논문 원고는 고치지 않았고, 측정값만 적는다.\n')
    w('- 입력 `1×3×224×224`, `model.eval()`, CPU, 무작위 초기화(가중치 불필요).')
    w('- 단위: Params는 M(10⁶), 연산량은 G(10⁹) MACs. 1 MAC = 곱셈-덧셈 1회.')
    w('- 수치는 정수 MACs 원값으로 계산한 뒤 소수 셋째 자리로 표시했다. 그래서 표시된 값끼리 더한 값과 끝자리가 1 다를 수 있다. '
      '원값은 [results/results.json](results/results.json)에 있다.\n')

    w('## 결과 표\n')
    w(table(['모델', 'Params (M)<br>논문 → 재측정', 'MACs (G)<br>논문 → 재측정(합계)', '(가) fvcore 기본', '(나) 노트북 처리 함수',
             '+(a)', '+(b)', '+(c)', '합계'], MAIN, main_row))
    w('')
    w('- **논문 → 재측정** 논문 보고 수치 → 이번 재측정 수치, 괄호 안은 차이(재측정 − 논문). '
      'MACs의 재측정 수치는 4절 규약의 합계 (가)+(a)+(b)+(c)이다. 논문 표기에 맞춰 재측정 수치를 반올림했다: '
      'Params는 소수 첫째 자리에서, MACs는 소수 둘째 자리에서 반올림. 차이도 반올림한 값끼리 계산했다. '
      '나머지 열은 반올림 전 값을 소수 셋째 자리까지 적었다.')
    w('- **(가)** fvcore 0.1.5 기본값. 노트북 6번 셀과 같다.')
    w('- **(나)** 노트북 7번 셀처럼 `rfft_flop_jit`을 `aten::fft_rfft2`, `aten::fft_irfft2`에 등록한 값. SPANet·SPANetV2만 적었다.')
    w('- **+(a), +(b), +(c)** (가)에 (a) FFT, (b) 주파수 영역의 원소별 곱, (c) 그 밖의 원소별 연산을 차례로 더한 누적값. '
      '**합계** = (가)+(a)+(b)+(c) 이므로 +(c)와 같다. 규약은 [4절](#4-하나의-규약으로-다시-잰-값)에 있다.')
    w(f"- Params는 `model.parameters()`의 원소 수 전체다. SPANet의 SPF 필터는 `requires_grad=False`인 `nn.Parameter`라서 포함된다 "
      f"(S {M(spanet_trainable['SPANet-S'])} M, M {M(spanet_trainable['SPANet-M'])} M, B {M(spanet_trainable['SPANet-B'])} M). "
      f"다른 모델은 학습 파라미터 수와 같다.")
    w('- SPANet-S / M / B는 크기가 가까운 S18 / S36 / M36 묶음의 pure 쪽에 두었다.\n')

    w('항목별 증가분 (MACs, 정수 원값). 마지막 열은 fvcore 무시 목록에 있어 어느 값에도 들어가지 않은 `aten::relu`의 출력 원소 수로, 참고용이다.\n')
    w(table(['모델', '(a) FFT', '(b) 주파수 영역 곱', '(c) 그 밖의 원소별', '(a)+(b)+(c)', '참고: relu 원소 수'], MAIN, increments_row))
    w('')

    # 1
    w('## 1. 환경\n')
    w(env_section())
    w('')
    w('- 모든 모델을 위의 한 파이썬 환경(`spanetv2` conda env)에서 쟀다. 저장소가 지정한 timm 버전은 0.6.11(SPANet, MetaFormer), '
      '0.6.13(SPANetV2), 0.5.4(DFFormer)이지만 0.6.13으로 모두 import 되었다.')
    w('- 지정 커밋 네 개는 측정일(2026-11-19)에 각 저장소 기본 브랜치의 HEAD와 같았다. GFNet은 지정 커밋이 없어 HEAD를 썼다.')
    w('- SPANetV2는 이 저장소의 `image_classification/models/spanetv2.py` 파일만 경로로 불러왔다. '
      '`models/__init__.py`를 거치지 않으므로 `spanetv2_ver4.py`는 import 되지 않는다.')
    w('- GFNet의 측정 파일 상태 `M`은 아래 import 수정 한 줄이다.\n')
    w('### import 수정\n')
    w('SPANet, SPANetV2, MetaFormer, DFFormer는 고친 줄이 없다. GFNet `gfnet.py` 7번째 줄만 고쳤다. numpy 2.x에서 `numpy.lib.arraypad`가 '
      '비공개 모듈이 되어 import가 실패했고, `pad`라는 이름은 파일 안에서 쓰이지 않는다. 계산 코드는 그대로다 '
      '([patches/GFNet_numpy_import.patch](patches/GFNet_numpy_import.patch)).\n')
    w('```diff\n-from numpy.lib.arraypad import pad\n+from numpy import pad  # import path fixed for numpy>=2 (was: from numpy.lib.arraypad import pad)\n```\n')

    # 2
    w('## 2. 기존 노트북 재현 (SPANet·SPANetV2)\n')
    w('각 저장소의 `image_classification/FLOPs_count.ipynb`를 그대로 따랐다(처리 함수인 2번 셀과 측정 코드인 6·7번 셀은 두 저장소에서 같고, 출력 문구만 다르다). '
      '7번 셀의 시간 측정 부분은 CUDA가 필요하고 연산량과 무관해 뺐다. `rfft_flop_jit`은 노트북 2번 셀을 그대로 복사했다.\n')
    w(table(['모델', '논문 MACs', '(가)', '(나)', '(가) 0.1 반올림', '(나) 0.1 반올림', '처리 함수 호출 수', '논문 값과 맞는 쪽'],
            spanet + spanetv2, notebook_row, groups=False))
    w('')
    m_ga = [n for n in spanetv2 if r1(R[n]['ga']) == R[n]['paper_macs_G']]
    m_na = [n for n in spanetv2 if r1(R[n]['na']) == R[n]['paper_macs_G']]
    m_none = [n for n in spanetv2 if n not in m_ga and n not in m_na]
    w(f'0.1G 단위로 반올림해 비교하면, SPANet 3개는 (가) = (나)라서 둘 다 논문 값과 같다. SPANetV2 {len(spanetv2)}개 중 (가)는 {len(m_ga)}개, '
      f'(나)는 {len(m_na)}개에서 논문 값과 같고, ' + ', '.join(m_none) + '는 둘 다 다르다.\n')
    w('- SPANet은 `torch.fft.fft2` / `ifft2`(`aten::fft_fft2`, `aten::fft_ifft2`)를 쓴다. 처리 함수는 `rfft2` / `irfft2`에만 등록되므로 '
      '한 번도 불리지 않고, (나) = (가)이다.')
    w(f"- SPANetV2는 `(B, C, H, W)` 텐서에 `dim=(2, 3)`으로 FFT를 한다. 처리 함수는 `inputs[0]`을 `B, H, W, C`로 풀기 때문에, "
      f"예를 들어 S18 stage 0의 `rfft2` 입력 {tuple(nb_rf['in_shape'])}은 H={nb_rf['unpacked']['H']}, W={nb_rf['unpacked']['W']}, "
      f"C={nb_rf['unpacked']['C']}로 읽혀 {int(nb_rf['value']):,}이 된다. 같은 호출을 4절 규약으로 세면 {rf['count']:,}이다. "
      f"`irfft2`에는 반쪽 스펙트럼 {tuple(nb_ir['in_shape'])} ({nb_ir['in_dtype']})이 들어가 {int(nb_ir['value']):,}이 된다. "
      f"모든 모델의 첫 호출은 아래 표에 있다.\n")
    w(nb_check_rows())
    w('')

    # 3
    w('## 3. fvcore 기본값에서 빠진 연산\n')
    w('모델마다 fvcore 기본값의 `unsupported_ops()`를 그대로 옮겼다. 모듈 이름은 그 연산을 직접 부르는 가장 안쪽 모듈이며, '
      '블록 번호 같은 숫자는 `*`로 묶고 클래스 이름과 호출 횟수를 붙였다. `unsupported_ops(모듈)`에서 자식 모듈 몫을 빼서 구했다.\n')
    w('전체 모델에서 나온 연산: ' + ', '.join(f'`{o}`' for o in ops_all) + '.\n')
    w('fvcore가 따로 무시하는 연산(`aten::relu`, `aten::to`, `aten::copy_`, `aten::permute` 등 `_IGNORED_OPS`)은 `unsupported_ops()`에 나오지 않는다. '
      'fvcore 기본값은 정규화 층(`layer_norm`, `group_norm`)은 센다(입력 원소 수 × 5, affine이 없으면 × 4).\n')
    for n in ORDER:
        w(f'<details><summary><b>{n}</b></summary>\n')
        w(unsupported_block(R[n]))
        w('\n</details>\n')

    # 4
    w('## 4. 하나의 규약으로 다시 잰 값\n')
    w('[measure.py](measure.py)의 `Convention` 클래스가 fvcore op handle을 새로 등록해 모든 모델에 똑같이 적용한다. '
      '노트북의 `rfft_flop_jit`은 쓰지 않는다. 등록한 처리 함수를 거치지 않은 연산이 하나라도 남으면 멈추게 했고, 모든 모델에서 남은 연산이 없었다. '
      '(가)+(a)+(b)+(c)는 처리 함수를 등록한 `FlopCountAnalysis.total()`과 모든 모델에서 같았다.\n')
    w('- **(a) FFT** — `fft_rfft2`, `fft_irfft2`, `fft_fft2`, `fft_ifft2`(및 `fftn`·1차원 변형). 변환 축은 연산의 `dim` 인자에서 읽는다. '
      '실수 영역 텐서의 원소 수를 E(`rfft*`는 입력, `irfft*`는 출력; 복소 변환 `fft*`/`ifft*`는 복소 출력), 변환 축 크기의 곱을 N이라 하고 '
      '한 번의 변환을 **E·⌈log2 N⌉** MACs로 센다(기존 노트북과 같은 규약; 순변환·역변환 한 쌍이면 2·H·W·C·⌈log2(H·W)⌉). '
      '`fft_fftshift`, `fft_ifftshift`는 0. 변환마다 실수/복소를 적었다.')
    w('- **(b) 주파수 영역의 원소별 곱** — 출력이 복소수인 `aten::mul`. 출력 원소 하나를 1로 센다. '
      'SPANet의 SPF 마스크 곱, SPANetV2의 SRF 곱, DFFormer·CDFFormer·GFNet의 `x * weight`가 여기에 든다.')
    w('- **(c) 그 밖의 원소별 연산** — fvcore 기본값이 세지 않은 나머지 산술 연산(`add`, `sub`, 실수 `mul`, `div`, `pow`, `sqrt`, `square`, '
      '`gelu`, `sigmoid`, `softmax`, `mean`). 출력 원소 하나를 1로 센다. `mean`도 규약대로 출력 원소 수로 셌고, 입력 원소 수는 아래 상세 표에 따로 남겼다.')
    w('- **0으로 센 연산** — 자료 이동·형 변환: `clone`, `real`, `view_as_complex`. fvcore 무시 목록의 `relu`는 그대로 0이며, 원소 수만 참고로 남겼다.')
    w('- 각 연산의 원소 수와 자료형(Float / ComplexFloat)을 남겼으므로 다른 규약(예: 복소 곱 = 실수 곱 4회)으로 다시 계산할 수 있다.\n')
    w('### 처리 함수 확인\n')
    w('모델마다 FFT 연산 종류별로 첫 호출 하나를 골라, 입력 shape, `dim` 인자, 계산된 값을 출력했다. '
      'ConvFormer·CAFormer에는 FFT 연산이 없다.\n')
    w(fft_check_rows())
    w('')
    w('### 모델별 (a)(b)(c) 상세\n')
    for n in ORDER:
        w(f'<details><summary><b>{n}</b></summary>\n')
        w(convention_block(R[n]))
        w('\n</details>\n')

    # 5
    w('## 5. 다른 도구로 교차 확인: `torch.utils.flop_counter.FlopCounterMode`\n')
    w('- torch 2.8.0 내장 도구다. 디스패치 단계에서 연산을 세며, 단위는 **FLOPs(곱셈-덧셈 1회 = 2)**이다. 그래서 2로 나눠 MACs로 비교했다.')
    w('- 세는 연산은 `flop_registry`에 등록된 `mm`, `addmm`, `bmm`, `baddbmm`, `convolution`, `_scaled_mm`, scaled-dot-product attention 계열뿐이다. '
      '**FFT, 원소별 곱, 정규화 층은 세지 않는다.** DFFormer의 복소 `einsum`은 `bmm`으로 분해되어 shape 기준으로 세어진다.')
    w('- DeepSpeed 프로파일러는 이 환경에 설치되어 있지 않아 쓰지 않았다.')
    w('- 아래 표의 `(가) − norm − FCM/2` 열은 (가)에서 fvcore가 센 정규화 층 몫과 FlopCounterMode/2를 뺀 값이다(정수 MACs). '
      'DFFormer·CDFFormer 외에는 0이다. 두 모델 계열에서 남는 수천 MACs 이하의 차이는 복소 `einsum`을 fvcore는 `np.einsum_path`의 추정값으로, '
      'FlopCounterMode는 `bmm` shape로 세서 생긴다.\n')
    w(table(['모델', '(가)', 'FCM FLOPs', 'FCM FLOPs / 2', '(가)의 norm 몫', '(가) − norm − FCM/2 (MACs)', 'FCM 연산별 FLOPs (G)'], ORDER, fcm_row))
    w('')

    # 6
    w('## 6. (선택) GFNet-H\n')
    w('GFNet 공식 저장소(`raoyongming/GFNet`)의 `GFNetPyramid`를 `infer.py` / `main_gfnet.py`의 `gfnet-h-s`, `gfnet-h-b` 인자 그대로 만들었다. '
      '공식 저장소에는 `gfnet-h-ti`, `gfnet-h-s`, `gfnet-h-b`만 있고 **GFNet-H-S++, GFNet-H-L++는 없어 재지 못했다.** '
      '저장소 README의 보고값은 H-S 32M / 4.5G, H-B 54M / 8.4G이다.\n')
    w(table(['모델', 'Params (M)<br>학위논문 표 → 재측정', 'MACs (G)<br>학위논문 표 → 재측정(합계)', '(가) fvcore 기본', '(나)',
             '+(a)', '+(b)', '+(c)', '합계'], GFNET, main_row, groups=False))
    w('')

    # diffs
    w('## 논문 값과의 차이\n')
    w('**MACs.** (가)와 논문 값의 차이는 모든 모델에서 0.1G 이하다. 차이가 큰 순서로 다섯 개:\n')
    for d, n in ga_diff[:5]:
        w(f'- {n}: (가) {G(R[n]["ga"])} − 논문 {R[n]["paper_macs_G"]} = {d:+.3f}')
    w('')
    if na_over:
        w('(나)는 다음 모델에서 논문 값과 0.1G 넘게 다르다.\n')
        for d, n in na_over:
            w(f'- {n}: (나) {G(R[n]["na"])} − 논문 {R[n]["paper_macs_G"]} = {d:+.3f}')
        w('')
    w('**Params.** 1M 단위로 반올림한 값이 논문 값과 다른 모델:\n')
    for n, p, pp in p_diff:
        w(f'- {n}: {M(p)} M (논문 {pp} M)')
    w('')
    w('### 설정 확인\n')
    w('위 모델들에 대해, 코드가 실제로 만든 설정을 모델 객체에서 읽어 출력했다([config_check.py](config_check.py), '
      '[results/config_check.txt](results/config_check.txt)). 채널 수, 블록 수, token mixer, ExSPAM head 수와 커널 집합, '
      'SRF 필터 크기, attention head 수, 구성 요소별 Params이다. 같은 폭의 ConvFormer·CAFormer도 함께 실었다. '
      '학위논문 원고가 이 폴더에 없어서, 원고의 설정표와의 대조는 하지 않았다.\n')
    w('SPANetV2 공통(코드 기준): ExSPAM `num_heads` 4, head별 depthwise 커널 3·5·7·9, SRF 필터 `(C/4, H, W//2+1)`(224 입력 기준 '
      '56×29, 28×15, 14×8, 7×4), proj 확장비 2, MLP 비 4(StarReLU), head `MlpHead`(비 4). hybrid는 stage 2 `MixAttention`(head_dim 32, 7×7 local conv), '
      'stage 3 `Attention`(head_dim 32). SPANet(비교용): stage마다 SPF 3개(λ 0.7, 0.8, 0.9), 1×7·7×1 depthwise conv, 반경은 코드 인자 `radius=[2, 2, 1, 1]`.\n')
    flagged = sorted({n for _, n in na_over} | {n for n, _, _ in p_diff}, key=ORDER.index)
    w(cfg_block(flagged + ['ConvFormer-M36', 'CAFormer-M36', 'ConvFormer-B36', 'CAFormer-B36']))
    w('')

    # files
    w('## 파일\n')
    w('| 파일 | 내용 |\n|---|---|')
    w('| [setup_third_party.sh](setup_third_party.sh) | 비교 저장소를 측정 커밋으로 `third_party/`에 clone 하고 GFNet import 수정 적용 |')
    w('| [zoo.py](zoo.py) | 모델 생성. 모델 파일을 경로로 불러온다(`models` 패키지 이름 충돌 방지) |')
    w('| [measure.py](measure.py) | 1·2·3·4·5단계 측정. `results/results.json`, `results/env.json` 생성 |')
    w('| [config_check.py](config_check.py) | 설정과 구성 요소별 Params 출력 → `results/config_check.txt` |')
    w('| [report.py](report.py) | 이 README 생성 |')
    w('| [patches/GFNet_numpy_import.patch](patches/GFNet_numpy_import.patch) | GFNet import 경로 수정 |')
    w('| [results/](results/) | 측정 원값(`results.json`), 환경(`env.json`), 실행 로그(`measure_log.txt`), 설정 출력 |')
    w('')
    w('## 재현\n')
    w('```bash\ncd flops_benchmark\nbash setup_third_party.sh\nconda activate spanetv2   # torch 2.8.0, fvcore 0.1.5.post20221221, timm 0.6.13\n'
      'python measure.py > results/measure_log.txt 2>&1\npython config_check.py > results/config_check.txt\npython report.py\n```')

    with open(os.path.join(HERE, 'README.md'), 'w') as f:
        f.write('\n'.join(md) + '\n')
    print('wrote README.md')


if __name__ == '__main__':
    main()
