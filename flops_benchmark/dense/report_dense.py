"""
Write README.md from results/ (measure_dense.py, repo_get_flops.py via run_step2.sh, config_check_dense.py).

    python report_dense.py
"""
import json
import os
import re
from decimal import ROUND_HALF_UP, Decimal

import measure_dense as D

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')


def load(name, default=None):
    p = os.path.join(RES, name)
    return json.load(open(p)) if os.path.exists(p) else default


R = {}
for t in ('det', 'seg', 'cls384'):
    R.update(load(f'{t}.json', {}))
ENV = {t: load(f'env_{t}.json') for t in ('det', 'seg', 'cls384')}
CC = {}
for t in ('det', 'seg'):
    CC.update(load(f'config_check_{t}.json', {}))
TOOL_ENV = {e: load(os.path.join('step2', f'env_{e}.json')) for e in ('spanetv2-dense', 'dfformer-mm0')}
SETTING_ORDER = ['mask_rcnn', 'cascade', 'upernet', 'fpn', 'cls384']
KEYS = [f'{m[0]}/{m[1]}' for m in D.MODELS if f'{m[0]}/{m[1]}' in R]
CAT = {'a': '(a) FFT', 'b': '(b) 주파수 영역 곱', 'c': '(c) 그 밖의 원소별', 'd': '(d) 마스크·필터 크기 조정',
       'zero': '0 (자료 이동·항등)', 'ignored': '0 (mmengine 무시 목록, 참고)', 'notcounted': '0 (세지 않음, 참고)'}


def G(x):
    return '-' if x is None else f'{x / 1e9:.3f}'


def Mp(x):
    return '-' if x is None else f'{x / 1e6:.3f}'


def pct(meas, ref):
    return (meas - ref) / ref * 100


def rounded(value, paper):
    """value (already in the paper's unit) rounded half up to the paper's number of decimals"""
    q = Decimal('1') if float(paper).is_integer() and not isinstance(paper, float) else Decimal('0.1')
    return Decimal(str(value)).quantize(q, rounding=ROUND_HALF_UP)


def keys_of(setting):
    return [k for k in KEYS if R[k]['setting'] == setting]


def paper_str(r):
    p, m = r['paper_params_M'], r['paper_macs_G']
    return f'{m} ({p})' if p is not None else f'{m}'


def table(header, rows):
    out = ['| ' + ' | '.join(header) + ' |', '|' + '|'.join(['---'] * len(header)) + '|']
    out += ['| ' + ' | '.join(str(c) for c in row) + ' |' for row in rows]
    return '\n'.join(out)


def grouped(keys, row_fn, ncol):
    rows, cur = [], None
    for k in keys:
        if R[k]['group'] != cur:
            cur = R[k]['group']
            rows.append([f'**{cur}**'] + [''] * (ncol - 1))
        rows.append(row_fn(R[k]))
    return rows


def link(path, text=None):
    return f'[{text or path}]({path})'


# ---------------------------------------------------------------------------------------------
MAIN_HEADER = ['모델', 'Params (M)', '논문 값', '(가) 기본', '+(a)', '+(b)', '+(c)', '+(d)', '합계', '그중 backbone']


def main_row(r):
    ga, a, b, c, d = r['ga'], r['a'], r['b'], r['c'], r['d']
    bb = G(r['backbone']['total']) if r.get('backbone') else '-'
    return [r['name'], Mp(r['params']), paper_str(r), G(ga), G(ga + a), G(ga + a + b), G(ga + a + b + c),
            G(ga + a + b + c + d), G(r['total']), bb]


def unused_note(keys):
    items = []
    for k in keys:
        r = R[k]
        u = [x for x in r.get('unused_modules', []) if x['params']]
        if u:
            items.append(f"{r['name']} " + ', '.join(f"`{x['module']}` ({x['cls']}) {Mp(x['params'])}M" for x in u))
    return items


def setting_section(s):
    st = D.SETTINGS[s]
    keys = keys_of(s)
    if not keys:
        return ''
    H, W = st['hw']
    out = [f"### {st['title']} — 입력 1×3×{H}×{W}", '']
    out.append(table(MAIN_HEADER, grouped(keys, main_row, len(MAIN_HEADER))))
    out.append('')
    notes = []
    if st['task'] == 'det':
        notes.append(f'모든 모델에서 RoI는 bbox head마다 {D.NUM_ROIS}개, mask head마다 그중 앞의 {D.NUM_MASK_ROIS}개로 셌다 '
                     '(2절 끝, 4절의 `TwoStageFixedRoI`). Params는 검출기 전체다.')
    if st['task'] == 'seg':
        aux = [f"{R[k]['name']} {Mp(R[k]['params_auxiliary_head'])}" for k in keys if R[k].get('params_auxiliary_head')]
        notes.append('get_flops.py처럼 auxiliary head를 떼고 쟀다. Params에도 들어 있지 않다'
                     + (f" (auxiliary head 파라미터, M: {', '.join(aux)})." if aux else '.'))
    if st['task'] == 'cls384':
        notes.append('분류 모델 전체를 1×3×384×384로 쟀다. SPANetV2는 train_384.py처럼 기본 input_size(224)로 만들었다. '
                     '"그중 backbone" 열은 해당하지 않는다.')
    un = unused_note(keys)
    if un:
        notes.append('Params에는 forward에서 한 번도 호출되지 않는 모듈의 파라미터도 들어 있다(측정 도구가 세는 그대로): '
                     + '; '.join(un) + '.')
    out += [f'- {n}' for n in notes]
    return '\n'.join(out)


# ---------------------------------------------------------------------------------------------
def paper_diff_rows():
    rows = []
    for s in SETTING_ORDER:
        for k in keys_of(s):
            r = R[k]
            pm, pp = r['paper_macs_G'] * 1e9, r['paper_params_M']
            dga, dt = pct(r['ga'], pm), pct(r['total'], pm)
            dp = pct(r['params'], pp * 1e6) if pp is not None else None
            flag = abs(dga) > 1 or (dp is not None and abs(dp) > 1)
            rows.append(dict(key=k, setting=s, r=r, dga=dga, dt=dt, dp=dp, flag=flag))
    return rows


def diff_table(rows):
    h = ['설정', '모델', '논문 MACs', '(가)', '(가) 반올림', '(가) 차이', '합계', '합계 차이',
         '논문 Params', 'Params', 'Params 반올림', 'Params 차이']
    body = []
    for x in rows:
        r = x['r']
        pp = r['paper_params_M']
        body.append([D.SETTINGS[x['setting']]['title'], r['name'], r['paper_macs_G'], G(r['ga']),
                     rounded(r['ga'] / 1e9, r['paper_macs_G']), f"{x['dga']:+.2f}%", G(r['total']), f"{x['dt']:+.2f}%",
                     pp if pp is not None else '-', Mp(r['params']),
                     rounded(r['params'] / 1e6, pp) if pp is not None else '-',
                     f"{x['dp']:+.2f}%" if x['dp'] is not None else '-'])
    return table(h, body)


# ---------------------------------------------------------------------------------------------
def parse_step2(s, name):
    p = os.path.join(RES, 'step2', f'{s}__{name}.log')
    if not os.path.exists(p):
        return None
    txt = open(p).read()
    m = re.search(r'^RESULT_JSON (.*)$', txt, re.M)
    out = dict(log=os.path.relpath(p, HERE), printed=[], ok=m is not None)
    for line in txt.splitlines():
        if re.match(r'^(Use size divisor|Input shape|Flops|Params):', line) or line.startswith('Use size divisor'):
            out['printed'].append(line.strip())
    if m:
        out.update(json.loads(m.group(1)))
    else:
        err = [ln for ln in txt.splitlines() if 'Error' in ln]
        out['error'] = err[-1] if err else 'no RESULT_JSON'
    return out


def roi_summary(events):
    """RoIs per extractor call for the first counted image (between the first two 'call' markers)."""
    calls, cur = [], None
    for e in events:
        if e[0] == 'call':
            cur = []
            calls.append(cur)
        elif e[0] == 'roi_extractor' and cur is not None:
            cur.append((e[3][0] if e[3] else None, e[2]))
    out = []
    for c in calls:
        bbox = [n for sz, n in c if sz == 7]
        mask = [n for sz, n in c if sz == 14]
        # each get_model_complexity_info call traces twice (FlopAnalyzer, ActivationAnalyzer): keep the first half
        half = len(bbox) // 2, len(mask) // 2
        out.append((bbox[:half[0]] or bbox, mask[:half[1]] or mask))
    return out


def fmt_rois(lst):
    if not lst:
        return '-'
    if len(set(lst)) == 1:
        return f'{lst[0]}' + (f' ×{len(lst)}' if len(lst) > 1 else '')
    return ', '.join(str(x) for x in lst)


def step2_table(s):
    keys = keys_of(s)
    is_det = D.SETTINGS[s]['task'] == 'det'
    h = ['모델', '도구', '출력 (그대로)', '반환값 (G)', 'Params (M)', '논문 값', '논문과 차이', '통합 측정 (가)']
    if is_det:
        h += ['이미지별 반환값 (G)', 'RoI: bbox head / mask head (단계마다)']
    body = []
    for k in keys:
        r = R[k]
        t = parse_step2(s, r['name'])
        if t is None:
            body.append([r['name'], '-', '(실행하지 않음)'] + [''] * (len(h) - 3))
            continue
        tool = {'det': 'object_detection/tools/analysis_tools/get_flops.py (mmengine)',
                'seg': 'semantic_segmentation/tools/analysis_tools/get_flops.py (mmengine)',
                'dfformer': 'dfformer/segmentation/tools/get_flops.py (mmcv 1.x)'}.get(t.get('task'), '?')
        if not t['ok']:
            body.append([r['name'], tool, f"실패: {t['error']}"] + [''] * (len(h) - 3))
            continue
        raw = t['raw']
        flops = sum(x['flops'] for x in raw) / len(raw)
        params = raw[-1]['params']
        if os.path.exists(os.path.join(RES, 'step2', f"{s}__{r['name']}.gpu_oom.log")):
            tool += '<br>(GPU 메모리 부족 → CPU에서 다시 실행)'
        row = [r['name'], tool, '<br>'.join(t['printed']), G(flops), Mp(params), r['paper_macs_G'],
               f"{flops / 1e9 - r['paper_macs_G']:+.3f} ({pct(flops, r['paper_macs_G'] * 1e9):+.2f}%)", G(r['ga'])]
        if is_det:
            row.append(', '.join(G(x['flops']) for x in raw))
            rs = roi_summary(t['events'])
            uniq = sorted(set((tuple(b), tuple(m)) for b, m in rs))
            row.append('<br>'.join(f'{fmt_rois(list(b))} / {fmt_rois(list(m))}' for b, m in uniq))
        body.append(row)
    return table(h, body)


# ---------------------------------------------------------------------------------------------
def unsupported_block(k):
    r = R[k]
    rows = [[x['op'], x['calls'], f"`{x['module']}`", x['cls']] for x in r['unsupported_by_module']]
    out = [f"<details><summary>{r['name']} — {D.SETTINGS[r['setting']]['title']}</summary>", '',
           '`FlopAnalyzer(model, inputs).unsupported_ops()` 출력:', '', '```', r['unsupported_ops_raw'], '```', '',
           table(['연산', '호출 수', '모듈 (번호는 *)', '모듈 클래스'], rows), '', '</details>', '']
    return '\n'.join(out)


def interp_args(args):
    """aten::upsample_bicubic2d.vec(input, output_size?, align_corners, scale_factors?); None = not a constant in the trace"""
    names = ['output_size', 'align_corners', 'scale_factors']
    return ', '.join(f'{n}={v}' for n, v in zip(names, args))


def check_rows(keys):
    rows = []
    for k in keys:
        r = R[k]
        for f in r['fft_checks']:
            rows.append([r['name'], f['op'], f"`{f['module']}`", f"{f['in_shape']} {f['in_dtype']}",
                         f"{f['out_shape']} {f['out_dtype']}", f"dim={f['dim']}, s={f['s']}",
                         f"E={f['E']:,} ({f['E_tensor']}), N={f['N']:,}, ⌈log2 N⌉={f['ceil_log2_N']}", f"{f['count']:,}"])
        for f in r['interp_checks']:
            rows.append([r['name'], f['op'], f"`{f['module']}`", f"{f['in_shape']} {f['in_dtype']}",
                         f"{f['out_shape']} {f['out_dtype']}", interp_args(f['args']),
                         f"출력 원소 수 {f['count']:,}", f"{f['count']:,}"])
        if not r['fft_checks'] and not r['interp_checks']:
            rows.append([r['name'], '(FFT·보간 연산 없음)', '', '', '', '', '', ''])
    return rows


def convention_block(k):
    r = R[k]
    rows = [[CAT.get(x['cat'], x['cat']), x['op'], x['dtype'], f"`{x['module']}`", x['calls'], f"{x['elements']:,}",
             f"{x['count']:,}"] for x in r['convention_records']]
    bb = r.get('backbone')
    lines = [f"<details><summary>{r['name']} — {D.SETTINGS[r['setting']]['title']}</summary>", '',
             f"(a) {r['a']:,} · (b) {r['b']:,} · (c) {r['c']:,} · (d) {r['d']:,} MACs"]
    if bb:
        lines.append(f"backbone만: (가) {bb['ga']:,} · (a) {bb['a']:,} · (b) {bb['b']:,} · (c) {bb['c']:,} · "
                     f"(d) {bb['d']:,} · 합계 {bb['total']:,} MACs; 전체 모델 측정의 backbone 몫 "
                     f"{r['total_by_module_top'].get('backbone', 0):,} MACs")
    lines += ['', table(['항목', '연산', 'dtype', '모듈 (번호는 *)', '호출 수', '원소 수', 'MACs'], rows), '', '</details>', '']
    return '\n'.join(lines)


# ---------------------------------------------------------------------------------------------
def env_section():
    e = ENV.get('det') or ENV.get('seg') or ENV.get('cls384')
    lines = ['## 1. 환경', '']
    if e:
        mmdet = (ENV.get('det') or {}).get('mmdet', '-')
        mmseg = (ENV.get('seg') or {}).get('mmseg', '-')
        lines += ['표의 모든 값 (통합 측정, `measure_dense.py`, conda env `spanetv2-dense`):', '',
                  table(['항목', '값'], [
                      ['날짜', e['date']], ['Python', e['python']], ['PyTorch', e['torch']], ['mmengine', e['mmengine']],
                      ['mmcv', e['mmcv']], ['mmdet', mmdet], ['mmseg', mmseg], ['fvcore', e['fvcore']],
                      ['timm', e['timm']], ['NumPy', e['numpy']],
                      ['장치', f"{e['device']} ({e['cpu']}, torch threads {e['torch_threads']})"],
                      ['OS', e['platform']], ['Python 경로', f"`{e['python_executable']}`"]]), '']
    rows = []
    for env_name, t in TOOL_ENV.items():
        if t:
            t = {k: ('-' if isinstance(v, str) and v.startswith('- (') else v) for k, v in t.items()}
            if env_name == 'spanetv2-dense':        # mmdet / mmseg: the copies inside this repository (PYTHONPATH)
                t['mmdet'] = (ENV.get('det') or {}).get('mmdet', '-').split(' (')[0] + ' (저장소 사본)'
                t['mmseg'] = (ENV.get('seg') or {}).get('mmseg', '-').split(' (')[0] + ' (저장소 사본)'
            rows.append([env_name, t['python'], t['torch'], t['mmcv'], t['mmengine'], t['mmdet'], t['mmseg'],
                         t['timm'], t['numpy'], t.get('cuda_runtime'), t['gpu']])
    if rows:
        lines += ['2절의 저장소 도구 실행 (`repo_get_flops.py`, GPU 0):', '',
                  table(['conda env', 'Python', 'PyTorch', 'mmcv', 'mmengine', 'mmdet', 'mmseg', 'timm', 'NumPy', 'CUDA',
                         'GPU, 드라이버'], rows), '',
                  '- `dfformer-mm0`은 DFFormer의 requirements.txt 버전(torch 1.12.1, mmcv-full 1.6.2, mmsegmentation 0.29.1, '
                  'timm 0.5.4)이다. DFFormer의 get_flops.py만 이 env에서 돌렸다. mmcv 1.x와 mmseg 0.x가 필요하기 때문이다.',
                  '- `spanetv2-dense`에서 mmdet·mmseg는 설치하지 않았다. 이 저장소에 들어 있는 사본(object_detection/mmdet 3.3.0, '
                  'semantic_segmentation/mmseg 1.2.2)을 PYTHONPATH로 썼다. 버전은 object_detection·semantic_segmentation의 '
                  'pyproject.toml과 INSTALL.md를 따랐다(torch 2.1.0 cu121, mmcv 2.1.0, timm 0.6.13, numpy<1.26). '
                  'Python은 pyproject.toml의 3.10이다.', '']
    if e:
        rows = []
        for key, v in e['repos'].items():
            rows.append([key, f"`{v['commit']}`", v['remote'], v['status'].replace('\n', '<br>')])
        lines += ['저장소 커밋 (측정 시점):', '', table(['저장소', '커밋', 'remote', '변경 (추적 파일)'], rows), '',
                  '- SPANetV2-official의 커밋은 이 폴더(flops_benchmark/)를 더한 d1713e5다. object_detection/, '
                  'semantic_segmentation/, image_classification/models/spanetv2.py는 지시의 1b2cfee와 같다 '
                  '(`git diff --stat 1b2cfee HEAD -- ...` 결과 없음).',
                  '- SPANet-official의 변경 한 건은 아래 import 수정이다.', '']
    patch = open(os.path.join(HERE, 'patches', 'SPANet_det_backbone_mm2_imports.patch')).read()
    lines += ['### import 수정', '',
              '계산 코드는 고치지 않았다. import가 깨진 파일은 하나다. SPANet-official/object_detection/mmdet/models/backbones/spanet.py'
              '(mmcv 1.x·mmdet 2.x용)이며, import 줄만 mmcv 2.x·mmengine·mmseg 1.x 경로로 바꿨다. '
              '`get_root_logger`는 mmdet 2.x의 함수 이름이다. mmengine에 같은 일을 하는 `MMLogger.get_current_instance`를 같은 이름에 '
              '묶었다. 이 이름은 사전 학습 가중치를 읽는 `init_weights` 안에서만 쓰이고, 측정에서는 불리지 않는다. '
              '수정본은 `patches/SPANet_det_backbone_mm2_imports.patch`이고, `setup.sh`가 적용한다.', '',
              '```diff', patch.rstrip(), '```', '',
              '- DFFormer의 segmentation/backbone/{dfformer,cdfformer}.py는 고치지 않고 import했다 '
              '(`from mmseg.models.builder import BACKBONES`가 mmseg 1.2.2에도 있다).',
              '- metaformer_baselines.py는 고치지 않았다(감싼 코드는 6절).', '']
    return '\n'.join(lines)


def step2_section():
    lines = ['## 2. 기존 방식 재현 — 각 저장소의 get_flops.py', '',
             '각 저장소의 스크립트를 고치지 않고 돌렸다(`run_step2.sh` → `repo_get_flops.py`). launcher가 더한 것은 다음뿐이다.',
             '- 난수 시드 0.',
             '- config 덮어쓰기 `model.backbone.init_cfg=None`. config의 체크포인트 파일이 이 서버에 없고, 가중치는 필요 없다.',
             '- 함수 감싸기. 스크립트가 부르는 `get_model_complexity_info`의 반환값(반올림 전 정수)을 기록한다.',
             '- 검출기 forward hook. RoI 추출기에 들어간 RoI 수를 기록한다.',
             '',
             '도구마다 따로 적을 점:',
             '- **검출(SPANetV2 mmdet 3.x):** get_flops.py에는 입력 크기 인자가 없다. val dataloader로 COCO val2017 이미지를 읽어 '
             '`--num-images`장의 평균을 낸다. 그래서 test pipeline의 Resize를 '
             '`scale=(800, 1280), keep_ratio=False`로 바꿔, 모든 이미지가 1280×800(H×W)으로 들어가게 했다. '
             '이미지 3장을 썼다(`--num-images 3`). 모델은 mmengine의 `tensor` 모드로 돈다. '
             'RPN이 실제로 낸 proposal이 RoI head에 들어가므로 RoI 수가 입력과 가중치에 따라 달라질 수 있다.',
             '- **분할(SPANetV2 mmseg 1.x):** `--shape 512 2048`(UperNet), `--shape 512 512`(Semantic FPN). '
             'ConvFormer, CAFormer, SPANet의 새로 만든 config(6절)도 이 스크립트로 쟀다.',
             '- **DFFormer(mmseg 0.x):** DFFormer 저장소의 segmentation/tools/get_flops.py는 mmcv 1.x의 '
             '`mmcv.cnn.get_model_complexity_info`를 쓴다. 이 함수는 trace가 아니라 module hook으로 센다. '
             '모듈 종류별로 정해진 수식만 쓰고, 함수 호출(matmul, FFT, einsum 등)과 SyncBatchNorm은 세지 않는다. '
             '이 env의 torch 1.12.1+cu116 cuFFT가 RTX 4090에서 `CUFFT_INTERNAL_ERROR`를 낸다. 그래서 모델은 스크립트대로 GPU에 두고, '
             '`torch.fft.rfft2/irfft2` 두 함수만 CPU에서 계산했다. 이 카운터는 module만 세므로 값은 같다.', '']
    for s in ('mask_rcnn', 'cascade', 'upernet', 'fpn'):
        if keys_of(s):
            lines += [f"### {D.SETTINGS[s]['title']}", '', step2_table(s), '']
    lines += ['- "반환값"은 카운터가 돌려준 정수를 10^9로 나눈 값이다(검출은 이미지 평균). "통합 측정 (가)"는 결과 표의 (가)다. '
              '검출에서는 RoI 설정이 다르다(아래).',
              f'- 결과 표의 검출 값은 모든 모델에 같은 RoI 설정으로 쟀다. RPN 3×3 conv까지 세고, bbox head마다 RoI {D.NUM_ROIS}개, '
              f'mask head마다 그중 앞의 {D.NUM_MASK_ROIS}개를 넣는다. mmdet 2.x `forward_dummy`와 같은 구조다. '
              'Mask R-CNN에서는 mmdet 3.x get_flops.py와 같다(StandardRoIHead의 tensor 모드가 mask head에 `rois[:100]`을 넣는다). '
              'Cascade Mask R-CNN에서는 다르다. mmdet 3.x의 CascadeRoIHead tensor 모드는 세 단계 mask head에 모든 RoI(위 표의 수)를 넣는다.',
              '']
    return '\n'.join(lines)


def convention_section():
    import inspect
    src_wrap = inspect.getsource(D.TwoStageFixedRoI)
    lines = ['## 4. 하나의 규약으로 다시 잼', '',
             '`../measure.py`의 `Convention` 처리 함수(분류 재측정과 같은 코드)를 mmengine의 `FlopAnalyzer`에 '
             '`set_op_handle`로 그대로 등록했다. 처리 함수의 인자 형식은 fvcore와 같다. (가)는 처리 함수 없이 '
             '`FlopAnalyzer(model, inputs).total()`이다. get_flops.py가 부르는 `get_model_complexity_info`의 flops가 이 값이다. '
             '합계 = (가) + (a) + (b) + (c) + (d). 처리 함수를 등록한 분석의 total과 같은지 모델마다 확인했다(assert).', '',
             '- **(a) FFT:** 한 번의 변환을 E·⌈log2 N⌉로 센다. E는 실수 영역 텐서의 원소 수다(rfft*는 입력, irfft*는 출력, '
             '복소 fft*/ifft*는 복소 출력). N은 `dim` 인자의 축 크기를 곱한 값이다. fftshift/ifftshift는 0이다.',
             '- **(b) 주파수 영역의 원소별 곱:** 복소 출력인 aten::mul, 출력 원소 하나에 1.',
             '- **(c) 그 밖의 원소별 연산:** measure.py의 목록(add, sub, mul, div, pow, sqrt, gelu, sigmoid, softmax, mean 등), 출력 원소 하나에 1.',
             '- **(d) 마스크·필터 크기 조정:** aten::upsample_bicubic2d, 출력 원소 하나에 1. '
             'SPANetV2는 SRF 마스크를 224 격자로 만들어 둔다. 입력이 만드는 격자가 다르면 ExSPAM.forward의 `resize_weight`가 '
             'bicubic으로 보간한다. DFFormer·CDFFormer의 분할 백본(`weight_resize=True`)은 `resize_complex_weight`가 '
             '복소 필터를 bicubic으로 보간한다. 크기가 같아도 보간한다. 이것도 (d)에 넣었다. '
             '헤드의 보간(FPN의 nearest, UperNet·FPNHead의 bilinear)은 mmengine 기본 처리 함수가 이미 센다. 그래서 (가)에 들어 있다.',
             '- 분류 재측정에 없던 연산을 이렇게 처리했다.',
             '  - `aten::log2` → (c). mmdet RoI 추출기가 RoI를 FPN 단계에 배정할 때 쓴다(원소 1000개).',
             '  - `aten::feature_dropout` → 0. eval 모드 Dropout2d로 항등이다. aten::dropout은 무시 목록에 있다.',
             '  - `aten::max_pool2d` → 0. mmdet FPN의 다섯째 출력 단계가 `F.max_pool2d(x, 1, stride=2)`다. '
             '창이 1×1이라 산술이 없다. 다른 창 크기가 나오면 측정을 멈추게 했다.',
             '  - `prim::PythonOp.RoIAlignFunction`(mmcv RoIAlign) → 세지 않음. 원소별 연산이 아니고 규약에 정의가 없다. '
             '호출 수와 출력 원소 수는 기록했다.',
             '  - `aten::relu`/`relu_`는 분류 때처럼 0이다(mmengine 무시 목록). 원소 수만 기록했다.',
             '- 위 목록에 없는 연산이 나오면 측정이 멈춘다. 모든 모델이 멈추지 않고 끝났다.', '',
             '검출기는 아래 forward로 쟀다. 서브모듈은 검출기 자신의 모듈이고, 이름도 같다(backbone, neck, rpn_head, roi_head). '
             f'RoI는 이미지 안의 고정 상자 {D.NUM_ROIS}개다(`make_rois`, 시드 0). 상자 값은 어떤 수도 바꾸지 않는다.', '',
             '```python', src_wrap.rstrip(), '```', '']
    lines += ['### 처리 함수 확인', '',
              '모델마다 FFT 연산의 종류별 첫 호출과 보간 연산의 첫 호출이다(전체 모델 측정). '
              '인자의 None은 trace에서 상수가 아닌 값이다(forward에서 H, W로 계산). 실제 크기는 출력 shape에 있다.', '']
    for s in SETTING_ORDER:
        keys = keys_of(s)
        if keys:
            lines += [f"**{D.SETTINGS[s]['title']}**", '',
                      table(['모델', '연산', '모듈', '입력 shape, dtype', '출력 shape, dtype', '인자', '계산', '값 (MACs)'],
                            check_rows(keys)), '']
    lines += ['### 모델별 (a)(b)(c)(d) 내역', '']
    for s in SETTING_ORDER:
        for k in keys_of(s):
            lines.append(convention_block(k))
    return '\n'.join(lines)


def config_section(diff_rows):
    flagged = [x for x in diff_rows if x['flag']]
    lines = ['## 5. 논문 값과 1% 넘게 다른 모델 — 설정 확인', '',
             '논문 값과 비교했다. 차이 = (재측정 − 논문) / 논문이다. "반올림"은 논문 값의 자릿수로 반올림한 값이다(half up). '
             'MACs는 (가), 즉 각 저장소 도구와 같은 기본 분석으로 비교했다. 합계는 논문이 세지 않은 연산까지 더한 값이라 '
             '판정에 쓰지 않고 차이만 적었다. (가)나 Params가 1%를 넘게 다르면 굵게 적고, 아래에 설정을 적었다.', '',
             diff_table([dict(x, r=dict(x['r'], name=(f"**{x['r']['name']}**" if x['flag'] else x['r']['name'])))
                         for x in diff_rows]), '']
    if not flagged:
        return '\n'.join(lines)
    lines += ['아래는 1%를 넘는 모델의 설정이다(`config_check_dense.py`, 만든 모델과 config에서 읽음). '
              'backbone (가)는 같은 입력으로 backbone만 잰 값을 연산자별로 나눈 것이다.', '']
    for s in SETTING_ORDER:
        fk = [x['key'] for x in flagged if x['setting'] == s]
        if not fk:
            continue
        st = D.SETTINGS[s]
        rows = []
        for k in fk:
            r, c = R[k], CC.get(k, {})
            bb = c.get('backbone', {})
            ops = (r.get('backbone') or {}).get('ga_by_operator') or r.get('ga_by_operator', {})
            opstr = ', '.join(f'{op} {v / 1e9:.1f}' for op, v in sorted(ops.items(), key=lambda kv: -kv[1]))
            if st['task'] == 'det':
                setting_str = (f"RoI {c.get('rois_counted')}; test_cfg rpn {c.get('test_cfg_rpn')}, "
                               f"rcnn max_per_img {c.get('test_cfg_rcnn_max_per_img')}<br>neck {c.get('neck')}<br>"
                               f"rpn {c.get('rpn_head')}<br>roi_head {c.get('roi_head')}")
            elif st['task'] == 'seg':
                setting_str = f"decode_head {c.get('decode_head')}" + (f"<br>neck {c['neck']}" if c.get('neck') else '') \
                    + f"<br>auxiliary_head (config) {c.get('auxiliary_head_in_config')}, 측정에서 제외"
            else:
                setting_str = '분류 모델 전체'
            bstr = (f"{bb.get('type')} depths {bb.get('depths')} dims {bb.get('dims')}<br>token mixer {bb.get('token_mixers')}"
                    + (f"<br>patch_dims {bb.get('patch_dims')}" if bb.get('patch_dims') else '')) if bb else '-'
            bparams = Mp(r.get('params_backbone')) if r.get('params_backbone') is not None else '-'
            unused = ', '.join(f"{kk[7:-7]} {Mp(v)}" for kk, v in bb.items() if kk.startswith('unused_') and v)
            rows.append([r['name'], 'x'.join(str(v) for v in r['input']), setting_str, bstr,
                         bparams + (f' (이 중 forward에 안 쓰임: {unused})' if unused else ''), opstr])
        lines += [f"### {st['title']}", '',
                  table(['모델', '입력', 'RoI·neck·head 설정', 'backbone', 'backbone Params (M)', 'backbone (가), 연산자별 G'], rows), '']
        if s == 'upernet':
            rows = []
            for k in keys_of(s):
                r, c = R[k], CC.get(k, {})
                c5 = c.get('channels512')
                if not c5:
                    continue
                pm = r['paper_macs_G'] * 1e9
                rows.append([r['name'], paper_str(r), c.get('decode_head', {}).get('channels'),
                             f"{Mp(r['params'])} / {Mp(r['params_with_auxiliary_head'])}",
                             f"{G(r['ga'])} ({pct(r['ga'], pm):+.1f}%)", G(r['total']),
                             f"{Mp(c5['params'])} / {Mp(c5['params_with_auxiliary_head'])}",
                             f"{G(c5['ga'])} ({pct(c5['ga'], pm):+.2f}%)", G(c5['total'])])
            if rows:
                lines += ['UperNet config(semantic_segmentation/local_configs/upernet/SPANetV2/, 커밋 37e7a83 하나에만 있음)의 '
                          '`decode_head.channels`는 320이다. 확인용으로 이 값만 512로 바꿔 한 번 더 쟀다. 512는 mmsegmentation의 '
                          'UperNet 기본 config(예: `_base_/models/upernet_swin.py`) 값이다. 다른 설정은 같다. '
                          '이 값은 결과 표에 쓰지 않았다.', '',
                          table(['모델', '논문 값 MACs (Params)', 'config channels', 'Params (aux 제외 / 포함)', '(가) (논문과 차이)', '합계',
                                 'channels=512 Params (aux 제외 / 포함)', 'channels=512 (가) (논문과 차이)', 'channels=512 합계'], rows), '',
                          '- "aux 포함"은 auxiliary head(FCNHead)까지 더한 파라미터 수다. get_flops.py와 결과 표는 aux를 뺀다.', '']
    return '\n'.join(lines)


def files_section():
    cfgs = sorted(os.path.join('configs', d, f) for d in ('det', 'seg') for f in os.listdir(os.path.join(HERE, 'configs', d)))
    lines = ['## 6. 새로 만든 config와 감싼 코드', '',
             '이 서버와 저장소에서 다음 config를 찾지 못했다. 그래서 SPANetV2 config에서 backbone만 바꿔 만들었다. '
             'neck, head, test_cfg, 입력 크기는 그대로다.',
             '- SPANetV2 Cascade Mask R-CNN S36(local_configs/SPANetV2에는 S18만 있다)',
             '- ConvFormer·CAFormer의 Mask R-CNN, Cascade Mask R-CNN, UperNet',
             '- SPANet-S/M의 Semantic FPN(SPANet-official에 분할 코드가 없다. README의 Semantic Segmentation 항목도 체크되지 않았고, '
             'semantic_segmentation 폴더가 없다)', '']
    rows = []
    for s_ in SETTING_ORDER:
        for k in keys_of(s_):
            r = R[k]
            src = {'repo': '저장소', 'constructed': '새로 만듦', 'zoo': '../zoo.py (분류)'}[r['source']]
            rows.append([D.SETTINGS[s_]['title'], r['name'], f"`{r['config']}`" if r.get('config') else '-', src,
                         r.get('backbone_type', '-'), '<br>'.join(r.get('overrides') or []) or '-'])
    lines += ['모델별 config(측정 스크립트는 모두 `measure_dense.py`, 2절은 `repo_get_flops.py`):', '',
              table(['설정', '모델', 'config (이 폴더 기준 경로)', '출처', 'backbone type', '측정 때 덮어쓴 값'], rows), '']
    for p in cfgs:
        lines += [f'<details><summary>{p}</summary>', '', '```python', open(os.path.join(HERE, p)).read().rstrip(), '```',
                  '', '</details>', '']
    lines += ['감싼 코드와 등록(`backbones.py`):', '', '```python', open(os.path.join(HERE, 'backbones.py')).read().rstrip(),
              '```', '']
    return '\n'.join(lines)


def main():
    diff_rows = paper_diff_rows()
    parts = [
        '# 검출·분할 모델 연산량 재측정 (SPANet·SPANetV2와 비교 모델)', '',
        '학위논문의 COCO 검출 표와 ADE20K 분할 표에 실린 MACs와 #params를 다시 쟀다. '
        '규약은 분류 재측정(`../README.md`, `../measure.py`)과 같다. 측정 결과만 적는다.', '',
        '- **측정 조건:** CPU, eval 모드, 무작위 초기화한 가중치(체크포인트를 읽지 않음), `torch.rand` 입력(시드 0), 배치 1.',
        '- **단위와 표기:** 값은 모두 MACs(곱셈-덧셈 1회 = 1)다. G = 10^9, M = 10^6이다. 소수 셋째 자리까지, 반올림하기 전 값이다.',
        '- **열의 뜻:** (가)는 mmengine 기본 분석이다. tools/analysis_tools/get_flops.py와 같은 함수다. '
        '+(a)부터는 (가)에 차례로 더한 값이다. "그중 backbone"은 같은 입력으로 backbone만 잰 합계다.',
        '- **실행:** `python measure_dense.py det|seg|cls384` → `python report_dense.py`.', '',
        '## 결과', '']
    for s in SETTING_ORDER:
        sec = setting_section(s)
        if sec:
            parts += [sec, '']
    parts += [env_section(), step2_section(), '## 3. 빠진 연산 (unsupported ops)', '',
              'mmengine 기본 분석에서 처리 함수가 없던 연산이다. 전체 모델 기준이다. 모듈 이름은 trace의 scope이고, '
              '같은 모양의 모듈은 번호를 `*`로 묶었다.', '']
    for s in SETTING_ORDER:
        for k in keys_of(s):
            parts.append(unsupported_block(k))
    parts += [convention_section(), config_section(diff_rows), files_section(),
              '## 7. 파일과 재현', '',
              '| 파일 | 내용 |', '|---|---|',
              '| `setup.sh` | third_party 받기, SPANet import 패치 적용, (`envs`) conda env 두 개 만들기 |',
              '| `backbones.py` | ConvFormer·CAFormer 감싸기, SPANet·DFFormer 분할 백본 등록 |',
              '| `configs/` | 새로 만든 config 14개 |',
              '| `measure_dense.py` | 결과 표의 모든 측정 → `results/{det,seg,cls384}.json`, `results/env_*.json`, `results/measure_*.log` |',
              '| `repo_get_flops.py`, `run_step2.sh` | 2절: 저장소 도구 실행 → `results/step2/*.log`, `results/step2/env_*.json` |',
              '| `config_check_dense.py` | 5절 → `results/config_check_{det,seg}.json` |',
              '| `tool_env.py` | env 버전 기록 |',
              '| `report_dense.py` | 이 README |',
              '| `patches/SPANet_det_backbone_mm2_imports.patch` | import 수정 |', '',
              '```bash',
              'bash setup.sh envs                     # third_party, 패치, conda env',
              'conda activate spanetv2-dense',
              'python measure_dense.py det && python measure_dense.py seg && python measure_dense.py cls384',
              'python config_check_dense.py det && python config_check_dense.py seg',
              'bash run_step2.sh <COCO dir>           # annotations/instances_val2017.json, val2017/ 가 있는 폴더',
              'python report_dense.py',
              '```', '']
    with open(os.path.join(HERE, 'README.md'), 'w') as f:
        f.write('\n'.join(parts))
    print('wrote README.md', sum(len(p) for p in parts))


if __name__ == '__main__':
    main()
