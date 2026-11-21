#!/usr/bin/env python
"""data/per_model/* (written by analyze_srf.py) -> report.md and data/contrast_e.csv."""
import csv
import json
import math
import os
import re
import subprocess
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')
REPO = os.path.dirname(os.path.dirname(HERE))
MODELS = ['S18pure', 'S36pure', 'S18hybrid', 'S36hybrid']
MODEL_NAME = dict(S18pure='SPANetV2-S18-pure', S36pure='SPANetV2-S36-pure', S18hybrid='SPANetV2-S18-hybrid',
                  S36hybrid='SPANetV2-S36-hybrid')
ORDER = ['IN1K', 'COCO/MaskRCNN', 'COCO/Cascade', 'ADE20K/FPN', 'ADE20K/UperNet']
SHOW = {'IN1K': 'IN1K', 'COCO/MaskRCNN': 'COCO Mask R-CNN', 'COCO/Cascade': 'COCO Cascade',
        'ADE20K/FPN': 'ADE20K FPN', 'ADE20K/UperNet': 'ADE20K UperNet'}
KS = [3, 5, 7, 9]
BANDS = ['DC', 'low', 'mid', 'high']
FILTER_RE = re.compile(r'^backbone\.stages\.(\d+)\.(\d+)\.token_mixer\.filter_([1-4])$')


# ------------------------------------------------------------------------------------------------ helpers
def fnum(x, nd=4, sign=False):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return '—'
    return f'{x:+.{nd}f}' if sign else f'{x:.{nd}f}'


def read_csv(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v) if k not in ('model', 'task', 'ckpt', 'quantity', 'band') else v
            except ValueError:
                pass
    return rows


def _cell(c):
    return str(c).replace('\\|', '|').replace('|', '\\|')


def table(header, rows):
    out = ['| ' + ' | '.join(_cell(h) for h in header) + ' |', '|' + '|'.join('---' for _ in header) + '|']
    out += ['| ' + ' | '.join(_cell(c) for c in r) + ' |' for r in rows]
    return '\n'.join(out)


def layer_id_spanetv2(name, max_layer_id):          # copy of get_layer_id_for_spanetv2 (layer_decay_optimizer_constructor.py)
    if name.startswith('backbone.stages'):
        stage_id, block_id = int(name.split('.')[2]), int(name.split('.')[3])
        return {0: 1, 1: 2}.get(stage_id, 3 + block_id // 3 if stage_id == 2 else max_layer_id)
    raise ValueError(name)


def stage_id_spanetv2(name, max_stage_id):          # copy of get_stage_id_for_spanetv2
    if name.startswith('backbone.stages'):
        return int(name.split('.')[2]) + 1
    raise ValueError(name)


def code_lr_scale(name, pw):
    """lr_scale the SPANetV2LearningRateDecayOptimizerConstructor assigns to parameter `name` (None: default ctor)."""
    if not pw:
        return 1.0
    num_layers = pw['num_layers'] + 2
    if 'layer_wise' in pw.get('decay_type', 'layer_wise'):
        lid = layer_id_spanetv2(name, pw['num_layers'])
    else:
        lid = stage_id_spanetv2(name, num_layers)
    return pw['decay_rate'] ** (num_layers - lid - 1)


DET = 'object_detection/local_configs/SPANetV2'
SEG = 'semantic_segmentation/local_configs'
REPO_CFG = {
    ('S18pure', 'COCO/MaskRCNN'): f'{DET}/mask-rcnn_spanetv2_s18_pure_3x.py',
    ('S18pure', 'COCO/Cascade'): f'{DET}/cascade-mask-rcnn_spanetv2_s18_pure_fpn-3x_coco.py',
    ('S18pure', 'ADE20K/FPN'): f'{SEG}/fpn/SPANetV2/fpn_spanet_s18_pure_ade20k-40k.py',
    ('S18pure', 'ADE20K/UperNet'): f'{SEG}/upernet/SPANetV2/upernet_spanet_s18_pure_ade20k-160k.py',
    ('S36pure', 'ADE20K/FPN'): f'{SEG}/fpn/SPANetV2/fpn_spanet_s36_pure_ade20k-40k.py',
    ('S36pure', 'ADE20K/UperNet'): f'{SEG}/upernet/SPANetV2/upernet_spanet_s36_pure_ade20k-160k.py',
    ('S18hybrid', 'COCO/MaskRCNN'): f'{DET}/mask-rcnn_spanetv2_s18_hybrid_3x.py',
    ('S18hybrid', 'COCO/Cascade'): f'{DET}/cascade-mask-rcnn_spanetv2_s18_hybrid_fpn-3x_coco.py',
    ('S18hybrid', 'ADE20K/FPN'): f'{SEG}/fpn/SPANetV2/fpn_spanet_s18_hybrid_ade20k-40k.py',
    ('S18hybrid', 'ADE20K/UperNet'): f'{SEG}/upernet/SPANetV2/upernet_spanet_s18_hybrid_ade20k-160k.py',
    ('S36hybrid', 'ADE20K/FPN'): f'{SEG}/fpn/SPANetV2/fpn_spanet_s36_hybrid_ade20k-40k.py',
    ('S36hybrid', 'ADE20K/UperNet'): f'{SEG}/upernet/SPANetV2/upernet_spanet_s36_hybrid_ade20k-160k.py',
}
CFG_FIELDS = [('backbone_type', 'backbone type'), ('drop_path_rate', 'drop_path_rate'),
              ('init_cfg', 'init_cfg'), ('constructor', 'constructor'), ('paramwise_cfg', 'paramwise_cfg'),
              ('optimizer', 'optimizer'), ('accumulative_counts', 'accumulative_counts'),
              ('train_batch_size', 'batch/GPU'), ('param_scheduler', 'param_scheduler'), ('train_cfg', 'train_cfg')]


def _norm(v):
    if isinstance(v, dict):
        return {k: _norm(x) for k, x in v.items() if not (k == 'checkpoint' and isinstance(x, str))} | (
            {'checkpoint': os.path.basename(v['checkpoint'])} if isinstance(v.get('checkpoint'), str) else {})
    if isinstance(v, list):
        return [_norm(x) for x in v]
    return v


def repo_cfg_diff(M):
    sys.path.insert(0, HERE)
    import analyze_srf as A
    rows = []
    for m, D in M.items():
        for l in D['tasks']:
            ck = D['ins']['checkpoints'][l].get('cfg')
            rel = REPO_CFG.get((m, l))
            if rel is None:
                rows.append([MODEL_NAME[m], SHOW[l], '(저장소에 config 없음)', '—'])
                continue
            repo = A.cfg_facts(A.load_cfg_file(os.path.join(REPO, rel)))
            diffs = []
            for k, name in CFG_FIELDS:
                a, b = _norm(repo.get(k)), _norm((ck or {}).get(k))
                if a != b:
                    diffs.append(f'{name}: 저장소 `{json.dumps(a, ensure_ascii=False)}` / 체크포인트 '
                                 f'`{json.dumps(b, ensure_ascii=False)}`')
            rows.append([MODEL_NAME[m], SHOW[l], f'`{rel}`', '<br>'.join(diffs) if diffs else '같음'])
    return ['저장소 config(`*/local_configs/...`)와 체크포인트에 저장된 학습 config의 대조. '
            f'항목은 {", ".join(n for _, n in CFG_FIELDS)}이다(init_cfg의 경로는 파일 이름만 비교). '
            '위 표와 이 보고서의 모든 계산은 체크포인트 쪽 값을 쓴다.', '',
            table(['모델', '체크포인트', '저장소 config', '차이'], rows), '']


def lr_check(c):
    opt, si = c.get('optimizer') or {}, c.get('wd_only_schedule') or {}
    f = next(iter((opt.get('filters') or {}).values()), None)
    saved = f['lr_at_save'] / f['lr_scale'] if f and f.get('lr_at_save') is not None else None
    sim = si.get('lr_after_last_scheduler_step')
    return [f'{saved:.4g}' if saved is not None else '—', f'{sim:.4g}' if sim is not None else '—']


def sched_text(ps):
    out = []
    for s in ps:
        t = s['type']
        unit = 'ep' if s.get('by_epoch', True) else 'it'
        if t == 'LinearLR':
            out.append(f"LinearLR {s['start_factor']:g}→1 ({s['begin']}–{s['end']} {unit})")
        elif t == 'MultiStepLR':
            out.append(f"MultiStep ×{s['gamma']:g} @ {s['milestones']} {unit}")
        elif t == 'PolyLR':
            out.append(f"Poly p={s['power']:g}, η_min={s.get('eta_min', 0):g} ({s['begin']}–{s['end']} {unit})")
        else:
            out.append(t)
    return '; '.join(out)


# ------------------------------------------------------------------------------------------------ load
def load():
    M = {}
    for m in MODELS:
        d = os.path.join(DATA, 'per_model', m)
        if not os.path.exists(os.path.join(d, 'inspect.json')):
            continue
        with open(os.path.join(d, 'inspect.json')) as f:
            ins = json.load(f)
        S = {}
        for r in read_csv(os.path.join(d, 'srf_summary.csv')):
            S[(r['task'], int(r['stage']), int(r['kernel']), r['quantity'], r['band'])] = r
        fits = {(r['task'], int(r['stage']), int(r['kernel'])): r for r in read_csv(os.path.join(d, 'fit_decay.csv'))}
        fitb = read_csv(os.path.join(d, 'fit_decay_by_block.csv'))
        ref = {(int(r['stage']), int(r['kernel'])): r for r in read_csv(os.path.join(d, 'in1k_reference.csv'))}
        wdp = os.path.join(d, 'wd_only_reference.csv')
        wd = read_csv(wdp) if os.path.getsize(wdp) else []
        bv = dict(np.load(os.path.join(d, 'band_values.npz')))
        labels = [l for l in ORDER if l in ins['labels']]
        M[m] = dict(ins=ins, S=S, fits=fits, fitb=fitb, ref=ref, wd=wd, bv=bv, labels=labels,
                    tasks=[l for l in labels if l != 'IN1K'], stages=ins['stages'])
    return M


# ------------------------------------------------------------------------------------------------ section 0
def sec0(M):
    L = ['## 0. 먼저 확인한 것', '']
    # 0.1 files
    L += ['### 0.1 체크포인트 파일', '',
          '모두 GitHub Release `DoranLyong/SPANetV2-official`에서 파일마다 새 디렉터리(`ckpts/<tag>/<파일 이름>/`)로 받았다 '
          '(`download_ckpts.py`).', '']
    rel_path = os.path.join(DATA, 'release_assets.json')
    rel = json.load(open(rel_path)) if os.path.exists(rel_path) else {}
    rows = []
    for m, D in M.items():
        for l in D['labels']:
            c = D['ins']['checkpoints'][l]
            tag = c['path'].split('/')[1]
            a = rel.get(f"{tag}/{c['file']}")
            if a is None:
                chk = '—'
            else:
                chk = '크기 ' + ('같음' if a['size'] == c['size_bytes'] else '**다름**') + ', sha256 ' + (
                    '(Release에 digest 없음)' if not a['digest'] else
                    ('같음' if a['digest'] == 'sha256:' + c['sha256'] else '**다름**'))
            load = 'weights_only=True' if c['weights_only'] else 'weights_only=False (아래 주)'
            lc = c.get('local_copy')
            local = '' if not lc else (' · 저장소 `image_classification/ckpt/` 사본과 sha256 ' +
                                       ('같음' if lc['sha256_equal'] else '**다름**'))
            rows.append([MODEL_NAME[m], SHOW[l], f"`{tag}/{c['file']}`", f"{c['size_bytes']:,}", f"`{c['sha256']}`",
                         chk, load + local])
    L += [table(['모델', '체크포인트', 'Release 파일', '크기 (bytes)', 'sha256', 'Release API 대조', 'torch.load'], rows), '',
          'Release API 대조는 GitHub API의 자산 크기·digest(`data/release_assets.json`)와 비교한 결과다. '
          'digest는 일부 자산에만 공개되어 있다.', '']
    errs = {c['weights_only_true_error'] for D in M.values() for c in D['ins']['checkpoints'].values()
            if not c['weights_only']}
    L += ['주: `torch.load(path, map_location="cpu", weights_only=True)`를 먼저 시도했다. IN1K 파일은 모두 그대로 읽혔다. '
          'COCO·ADE20K 파일(mmengine 저장본)은 모두 다음 오류로 실패했다.', '']
    L += [f'> `{e}`' for e in sorted(errs)]
    L += ['', '오류는 학습 메타데이터(`message_hub`의 `HistoryBuffer`) 때문이다. 저자 본인의 공개 체크포인트이므로 '
          '`weights_only=False`로 읽었다.', '']
    rows = []
    for m, D in M.items():
        for l in D['labels']:
            c = D['ins']['checkpoints'][l]
            tk = c['top_level_keys']
            tk = ', '.join(f"'{k}'" for k in tk) if isinstance(tk, list) else tk
            shp = ' · '.join(f"s{s}: {tuple(v[0])}×{c['blocks_by_stage'][s]}" for s, v in c['filter_shapes_by_stage'].items())
            rows.append([MODEL_NAME[m], SHOW[l], tk, c['state_dict_used'], f"`{c['filter_key_prefix']}`",
                         c['n_filter_keys'], shp, '예' if c['same_filter_keys_and_shapes_as_IN1K'] else '**아니오**'])
    L += ['SRF key는 정규식 `filter_[1-4]$`로 찾았다. 이름 형식은 `<접두>stages.<s-1>.<block>.token_mixer.filter_<h>`이고, '
          '같은 블록의 커널은 `local_conv_<h>.weight`, shape (C/4, 1, k, k), k = 3, 5, 7, 9이다. '
          'shape 열은 (C/4, H, W//2+1)×블록 수이다.', '']
    L += [table(['모델', '체크포인트', '최상위 key', '쓴 state_dict', 'SRF key 접두', 'SRF key 수', 'SRF shape (단계별)',
                 'IN1K와 key·shape 같음'], rows), '']

    # 0.2 EMA / what init_weights loads
    L += ['### 0.2 IN1K 가중치: EMA 여부와 하류 학습이 실제로 불러온 쪽', '']
    flat = all(isinstance(D['ins']['checkpoints']['IN1K']['top_level_keys'], str) for D in M.values())
    ema = any(D['ins']['checkpoints']['IN1K']['has_state_dict_ema'] for D in M.values())
    L += [f"- IN1K 파일 {len(M)}개는 {'모두 평면 state_dict' if flat else '평면 state_dict가 아닌 것이 있음'}이다. "
          "최상위에 'state_dict', 'model', 'state_dict_ema' 같은 묶음 key가 없다. "
          f"EMA 가중치가 따로 들어 있지 {'않다' if not ema else '있다(확인 필요)'}. "
          "파일 안의 텐서 한 벌이 곧 IN1K 기준이다.",
          '- 하류 백본의 불러오기 경로(코드 확인):',
          '  - `SPANetV2.__init__`은 `init_cfg`가 있으면 `self.init_weights(pretrained)`를 직접 부른다 '
          '(object_detection/mmdet/models/backbones/spanetv2.py:556, semantic_segmentation/mmseg/models/backbones/spanetv2.py:555).',
          '  - `init_weights`는 `ckpt["state_dict"]` → `ckpt["model"]` → `ckpt` 순서로 고른다(같은 파일 585–590 / 584–589행). '
          '그리고 `load_state_dict(state_dict, False)`로 넣는다(594 / 593행).',
          "  - mmengine `BaseModule.init_weights`는 자식 모듈의 `init_weights()`를 호출한다(mmengine/model/base_module.py:136). "
          "SPANetV2가 이 메서드를 덮어쓰므로 백본에는 mmengine `PretrainedInit`이 아니라 위 함수가 적용된다.",
          '  - 따라서 IN1K 파일 전체(평면 dict)가 그대로 쓰인다. EMA를 고를 분기는 없다.',
          '- 각 하류 체크포인트에 저장된 학습 config(`meta["cfg"]`)의 `model.backbone.init_cfg`는 다음과 같다.', '']
    rows = []
    for m, D in M.items():
        in1k = D['ins']['checkpoints']['IN1K']['file']
        for l in D['tasks']:
            c = D['ins']['checkpoints'][l]
            cfg = c.get('cfg', {})
            ic = cfg.get('init_cfg') or {}
            ok = os.path.basename(str(ic.get('checkpoint'))) == in1k
            rows.append([MODEL_NAME[m], SHOW[l], f"`{cfg.get('backbone_type')}`", f"`{ic.get('type')}`",
                         f"`{ic.get('checkpoint')}`", '같음' if ok else '**다름**', f"`{cfg.get('load_from')}`"])
    L += [table(['모델', '체크포인트', 'backbone type', 'init_cfg.type', 'init_cfg.checkpoint', 'IN1K 파일 이름과', 'load_from'],
                rows), '']
    L += ['저장소에는 S36 Cascade config(`cascade-mask-rcnn_spanetv2_s36_*`)가 없다. 그래서 위 표의 S36 Cascade 행은 체크포인트 안의 config로 확인했다. '
          '체크포인트별 config 전문은 `data/per_model/<모델>/cfg_<과제>.py`에 저장했다.', '']

    # 0.3 optimizer
    L += ['### 0.3 하류 학습에서 filter가 받은 weight decay와 학습률 배율', '',
          '코드 규칙(object_detection/mmdet/engine/optimizers/layer_decay_optimizer_constructor.py, '
          'semantic_segmentation/mmseg/engine/optimizers/layer_decay_optimizer_constructor.py의 '
          '`SPANetV2LearningRateDecayOptimizerConstructor.add_params`):', '',
          '- `requires_grad=False`면 건너뛴다(det 144행, seg 165행).',
          "- no_decay 조건은 `len(param.shape) == 1 or name.endswith('.bias') or name in ('pos_embed', 'cls_token')`이다(det 146행, seg 167행). "
          'filter는 3차원 `(C/4, H, W//2+1)`이므로 `decay` 그룹이고 weight decay = config의 `weight_decay`다.',
          "- 백본에 `no_weight_decay()`가 `{'norm'}`을 돌려주는 skip list가 있다(det spanetv2.py:603). "
          '하지만 mmdet·mmseg 사본의 어디에서도 이 함수를 부르지 않는다(grep 결과 0건). 따라서 이 skip list는 적용되지 않는다.',
          '- layer_wise일 때 `get_layer_id_for_spanetv2(name, num_layers)`: stages.0 → 1, stages.1 → 2, '
          'stages.2 → 3 + block//3, stages.3 → num_layers.',
          '- stage_wise일 때 `get_stage_id_for_spanetv2(name, num_layers+2)`: stages.i → i+1.',
          '- 배율 = `decay_rate ** (num_layers + 2 − layer_id − 1)`(det 169행, seg 191행).',
          '- `paramwise_cfg`가 없으면(FPN config) mmengine 기본 constructor가 `model.parameters()` 전체를 한 그룹으로 만든다 '
          '(default_constructor.py:310). 이때 배율 1, weight decay = config 값이다.', '',
          '아래 표의 lr_scale과 wd는 체크포인트에 저장된 optimizer `param_groups`에서 filter 파라미터가 속한 그룹을 직접 읽은 값이다. '
          '"코드 규칙"은 위 규칙을 그 config에 적용해 다시 계산한 값이다. a_wd만은 저장된 일정(param_scheduler, 총 iteration, '
          '`accumulative_counts`)을 mmengine scheduler로 그대로 재현해 계산했다. '
          'AdamW의 decoupled decay(`p ← p·(1 − lr·wd)`)만 작용했을 때 M에 남는 배율 Π(1 − lr_k·wd)이다.', '']
    rows = []
    for m, D in M.items():
        for l in D['tasks']:
            c = D['ins']['checkpoints'][l]
            cfg, meta, opt = c.get('cfg', {}), c.get('meta', {}), c.get('optimizer') or {}
            pw = cfg.get('paramwise_cfg')
            o = cfg.get('optimizer', {})
            tc = cfg.get('train_cfg') or {}
            length = (f"{meta.get('epoch')} ep / {meta.get('iter'):,} it" if 'max_epochs' in tc
                      else f"{meta.get('iter'):,} it")
            rows.append([MODEL_NAME[m], SHOW[l], cfg.get('constructor', '').replace('OptimWrapperConstructor', ''),
                         (f"{pw['decay_type']}, {pw['decay_rate']}, {pw['num_layers']}" if pw else '— (기본)'),
                         f"{o.get('type')} {o.get('lr'):g}", f"{o.get('weight_decay'):g}", cfg.get('accumulative_counts'),
                         cfg.get('train_batch_size'), sched_text(cfg.get('param_scheduler') or []),
                         f"{tc.get('max_epochs', '')} ep" if 'max_epochs' in tc else f"{tc.get('max_iters', ''):,} it",
                         length, ', '.join(str(s) for s in opt.get('adam_steps', [])),
                         c.get('wd_only_schedule', {}).get('n_opt_steps_simulated', '—'), *lr_check(c)])
    L += [table(['모델', '체크포인트', 'constructor', 'decay_type, rate, num_layers', 'optimizer, base lr', 'weight_decay',
                 'accum', 'batch/GPU', 'param_scheduler', 'config 길이', '저장 시점 (meta)', 'Adam step (저장값)',
                 '재현한 optimizer step 수', '저장된 lr ÷ lr_scale', '재현한 저장 시점 lr'], rows), '',
          '마지막 두 열은 일정 재현을 검증하는 값이다. 저장된 filter 그룹의 lr을 lr_scale로 나눈 값과, 같은 iteration까지 '
          'scheduler를 돌린 lr을 나란히 적었다.', '']
    L += ['배치 크기는 GPU 하나당 값이다. GPU 수는 체크포인트에 기록되어 있지 않다.', '']
    L += repo_cfg_diff(M)
    early, gaps = [], []
    for m, D in M.items():
        for l in D['tasks']:
            c = D['ins']['checkpoints'][l]
            tc, meta = (c.get('cfg') or {}).get('train_cfg') or {}, c.get('meta') or {}
            if 'max_epochs' in tc and meta.get('epoch', 0) < tc['max_epochs']:
                early.append(f"{MODEL_NAME[m]} {SHOW[l]} ({meta['epoch']}/{tc['max_epochs']} ep)")
            if 'max_iters' in tc and meta.get('iter', 0) < tc['max_iters']:
                early.append(f"{MODEL_NAME[m]} {SHOW[l]} ({meta['iter']:,}/{tc['max_iters']:,} it)")
            st = (c.get('optimizer') or {}).get('adam_steps') or []
            sim = (c.get('wd_only_schedule') or {}).get('n_opt_steps_simulated')
            if st and sim:
                aw = [r['a_wd_only'] for r in D['wd'] if r['task'] == l]
                gaps.append(dict(name=f'{MODEL_NAME[m]} {SHOW[l]}', gap=(sim - st[0]) / sim, adam=st[0], sim=sim,
                                 meta=meta, aw=(min(aw), max(aw)) if aw else None, ratio=st[0] / sim))
    if early:
        L += ['config 끝보다 앞선 시점에 저장된 체크포인트(Release 파일이 그 시점 것): ' + '; '.join(early) + '.', '']
    if gaps:
        small = [g for g in gaps if g['gap'] < 0.01]
        big = [g for g in gaps if g['gap'] >= 0.01]
        if small:
            L += [f'저장된 Adam step은 재현한 optimizer step 수보다 {100 * min(g["gap"] for g in small):.3f}–'
                  f'{100 * max(g["gap"] for g in small):.3f}% 적다({len(small)}개 체크포인트). '
                  'AmpOptimWrapper(동적 loss scale)는 기울기에 inf/NaN이 있으면 그 step을 건너뛰는데, 이 차이는 그런 경우와 맞는다. '
                  'a_wd는 재현한 step 전체로 계산했다.', '']
        for g in big:
            ep, it = g['meta'].get('epoch'), g['meta'].get('iter')
            adj = '' if not g['aw'] else (
                f' Σlr를 이 비율({g["ratio"]:.4f})로 줄여 다시 계산하면 a_wd는 {g["aw"][0] ** g["ratio"]:.4f}–'
                f'{g["aw"][1] ** g["ratio"]:.4f}이다(재현값 그대로는 {g["aw"][0]:.4f}–{g["aw"][1]:.4f}).')
            L += [f'**예외 — {g["name"]}:** 저장된 Adam step {g["adam"]:,}은 재현한 {g["sim"]:,}보다 {100 * g["gap"]:.2f}% 적다. '
                  f'meta의 iter {it:,}가 epoch {ep}로 나누어떨어지지 않으므로(iter/epoch = {it / ep:,.2f}), '
                  'iteration 기반 재현이 실제 optimizer step 수와 맞지 않는 체크포인트다. 보고서의 a_wd는 재현값을 그대로 쓴다.' + adj, '']
    rows = []
    for m, D in M.items():
        wd_by = defaultdict(list)
        for r in D['wd']:
            wd_by[(r['task'], int(r['stage']))].append(r)
        for l in D['tasks']:
            c = D['ins']['checkpoints'][l]
            opt = c.get('optimizer') or {}
            pw = (c.get('cfg') or {}).get('paramwise_cfg')
            per_stage = defaultdict(list)
            for n, r in (opt.get('filters') or {}).items():
                s = int(FILTER_RE.match(n).group(1)) + 1
                per_stage[s].append((n, r))
            for s in sorted(per_stage):
                items = per_stage[s]
                sc = sorted({round(r['lr_scale'], 10) for _, r in items})
                wds = sorted({r['weight_decay'] for _, r in items})
                code = sorted({round(code_lr_scale(n, pw), 10) for n, _ in items})
                same = all(abs(code_lr_scale(n, pw) - r['lr_scale']) < 1e-9 for n, r in items)
                aw = [r['a_wd_only'] for r in wd_by[(l, s)]]
                groups = sorted({r['group'] for _, r in items if r['group']})
                rows.append([MODEL_NAME[m], SHOW[l], s, ', '.join(f'{x:.4g}' for x in sc),
                             ', '.join(f'{x:g}' for x in wds), ', '.join(f'{x:.4g}' for x in code),
                             '같음' if same else '**다름**', ', '.join(groups) if groups else '(그룹 1개)',
                             f'{min(aw):.4f}' + (f'–{max(aw):.4f}' if max(aw) - min(aw) > 5e-5 else '') if aw else '—'])
    L += [table(['모델', '체크포인트', '단계', 'filter lr_scale (저장값)', 'filter wd (저장값)', '코드 규칙 lr_scale',
                 '저장값 = 코드 규칙', 'param group', 'a_wd (decay만의 잔존 배율)'], rows), '']
    L += ['단계 안에서 값이 여러 개인 것은 블록마다 layer_id가 다르기 때문이다(layer_wise의 stages.2는 3 + block//3).', '']

    # 0.4 trained or frozen
    L += ['### 0.4 filter가 하류 학습에서 실제로 바뀌었는가', '']
    rows = []
    for m, D in M.items():
        for l in D['tasks']:
            c = D['ins']['checkpoints'][l]
            opt = c.get('optimizer') or {}
            fb = [r for r in D['fitb'] if r['task'] == l]
            zero = sum(1 for r in fb if r['rel_change'] == 0)
            rc = [D['fits'][(l, s, k)]['rel_change'] for s in D['stages'] for k in KS]
            mx = [D['fits'][(l, s, k)]['max_abs_dM'] for s in D['stages'] for k in KS]
            ea = [r['exp_avg_abs_mean'] for r in (opt.get('filters') or {}).values() if r.get('exp_avg_abs_mean') is not None]
            rows.append([MODEL_NAME[m], SHOW[l], f"{len(opt.get('filters') or {})} / {c['n_filter_keys']}",
                         len(opt.get('filters_not_in_optimizer') or []), f'{zero} / {len(fb)}',
                         f'{min(rc):.3f} – {max(rc):.3f}', f'{max(mx):.3f}',
                         f'{min(ea):.2e} – {max(ea):.2e}' if ea else '—'])
    L += [table(['모델', '체크포인트', 'optimizer에 있는 filter / 전체', 'optimizer에 없는 filter', 'ΔM = 0인 텐서 (블록·head)',
                 '‖ΔM‖/‖M_IN1K‖ 범위 (단계·head)', 'max |ΔM|', '저장된 Adam mean|exp_avg| 범위'], rows), '']
    L += ['- **동결 여부:** constructor는 `requires_grad=False`인 파라미터를 optimizer에 넣지 않는다. 모든 filter가 optimizer 그룹 안에 있으면 동결되지 않은 것이다. '
          'SPANetV2 백본과 위 config에는 `frozen_stages` 같은 동결 설정이 없다.',
          '- **변화량 기준:** ‖·‖는 반쪽 격자를 켤레 대칭 가중으로 전체 격자처럼 계산한 Frobenius 노름이다.', '']
    return L


# ------------------------------------------------------------------------------------------------ section 1
def sec1_conv(M):
    any_m = next(iter(M.values()))
    rows = []
    seen = set()
    for D in M.values():
        for s, g in D['ins']['grid'].items():
            if s in seen:
                continue
            seen.add(s)
            bw = g['band_cells_full_grid']
            rows.append([s, f"{g['H']}×{g['W']} (저장 {g['H']}×{g['Wr']})", g['n_ring'],
                         *(int(bw[b]) for b in BANDS)])
    rows.sort()
    return ['## 1. 측정', '',
            '### 1.0 규약', '',
            '- **주파수 격자:** ky = fftfreq(H), kx = rfftfreq(W) [cycles/pixel], ρ = √(ky² + kx²) / 0.5이다(축 방향 Nyquist = 1, 모서리 √2). '
            '224 입력의 저장 격자를 쓰므로 H = W다.',
            '- **반쪽 격자 가중:** 주파수 평균은 반쪽 격자를 켤레 대칭으로 펼친 전체 격자 평균이다. kx = 0 열과, W가 짝수일 때 Nyquist 열은 가중 1, 나머지 열은 가중 2다. '
            '전체 격자 FFT와 같은지는 단위 시험으로 확인했다.',
            '- **대역:** DC ρ = 0 | low 0 < ρ ≤ 0.25 | mid 0.25 < ρ ≤ 0.5 | high ρ > 0.5.',
            '- **단계 4의 low 대역:** 7×7에서 가장 작은 0이 아닌 ρ는 2/7 = 0.286이다. 그래서 low 대역이 비고 ψ_low와 high/low는 정의되지 않는다(표에서 —).',
            '- **반경 profile:** 정수 주파수 반경 r = round(√(iy² + ix²))의 고리로 나눈다(고리 중심 ρ = 2r/H).',
            '- **"채널 간" 통계:** 단계 안의 모든 블록을 모아 (블록, 채널) 쌍 n = 블록 수 × C/4개를 모집단으로 삼는다. '
            '평균은 채널 평균, p25·p75는 그 모집단의 백분위다. CSV의 `high/low` 행은 mean = mean(ψ_high)/mean(ψ_low), '
            'p25·p75 = 채널별 비의 백분위다.',
            '- **|K̂|:** `local_conv_h.weight`의 DTFT를 같은 (ky, kx)에서 계산했다. k ≤ H이면 H×W로 zero-pad한 rfft2와 같다. '
            '단계 4의 k = 9(9 > 7)는 커널을 H로 감아 접은 rfft2와 같다. bias는 넣지 않았다. '
            '"norm"이 붙은 양은 채널마다 전 주파수 평균으로 나눈 뒤 평균한 값이다.',
            '- **IN1K 대비 변화:** Δψ = ψ_task − ψ_IN1K. ‖ΔM‖/‖M_IN1K‖는 단계·head 안의 모든 (블록, 채널, 주파수)를 묶어 계산했다.',
            '- **weight decay 대조:** a = ⟨M_task, M_IN1K⟩ / ⟨M_IN1K, M_IN1K⟩(같은 가중 내적), R = M_task − a·M_IN1K다. '
            '"residual"은 σ(a·M_IN1K + R) − σ(a·M_IN1K) = ψ_task − σ(a·M_IN1K)이다. '
            '"decay_part"는 σ(a·M_IN1K) − ψ_IN1K이다(Δψ = decay_part + residual).', '',
            '격자별 대역 칸 수(전체 격자 기준):', '',
            table(['단계', '격자', '고리 수', 'DC', 'low', 'mid', 'high'], rows), '']


def sec1a(M):
    L = ['### 1.1 (a) 마스크 ψ = σ(M): 대역 평균', '',
         '- **값:** 각 칸은 (블록, 채널) 평균이다. Δ는 같은 단계·head의 IN1K 대비 차이다.',
         '- **채널 분포:** 채널 간 p25·p75는 `data/srf_summary.csv`(quantity = psi)에 있다.',
         '- **그림:** `figs/psi_radial_<모델>.png`(반경 profile, 회색 띠 = IN1K p25–p75), `figs/dpsi_radial_<모델>.png`(Δψ), '
         '`figs/psi_map_<모델>.png`(2D 지도).', '']
    for m, D in M.items():
        S = D['S']
        L += [f'#### {MODEL_NAME[m]}', '',
              f'![psi radial](figs/psi_radial_{m}.png)', '', f'![dpsi radial](figs/dpsi_radial_{m}.png)', '',
              f'![psi map](figs/psi_map_{m}.png)', '']
        rows = []
        for s in D['stages']:
            for k in KS:
                base_ratio = S[('IN1K', s, k, 'psi', 'high/low')]['mean']
                for l in D['labels']:
                    g = lambda b, q='psi': S[(l, s, k, q, b)]['mean']
                    ratio = g('high/low')
                    if l == 'IN1K':
                        d = ['', '', '']
                    else:
                        d = [fnum(g('low', 'delta_psi'), 4, True), fnum(g('high', 'delta_psi'), 4, True),
                             fnum(ratio - base_ratio, 4, True)]
                    rows.append([s, k, SHOW[l], fnum(g('DC')), fnum(g('low')), fnum(g('mid')), fnum(g('high')),
                                 fnum(ratio), *d])
        L += [table(['단계', 'k', '체크포인트', 'ψ_DC', 'ψ_low', 'ψ_mid', 'ψ_high', 'ψ_high/ψ_low', 'Δψ_low', 'Δψ_high',
                     'Δ(high/low)'], rows), '']
    return L


def sec1b(M):
    L = ['### 1.2 (b) 커널 |K̂|와 실효 응답 ψ·|K̂|', '',
         '- **값:** 채널마다 전 주파수 평균으로 나눈 값(norm)의 (블록, 채널) 평균이다.',
         '- **정규화 전 값:** 정규화하지 않은 |K̂|·ψ|K̂|와 p25·p75는 `data/srf_summary.csv`(quantity = kernel, effective, '
         'kernel_norm, effective_norm)에 있다.',
         '- **그림:** `figs/effective_radial_<모델>.png`, `figs/kernel_radial_<모델>.png`.', '']
    for m, D in M.items():
        S = D['S']
        L += [f'#### {MODEL_NAME[m]}', '', f'![effective radial](figs/effective_radial_{m}.png)', '',
              f'![kernel radial](figs/kernel_radial_{m}.png)', '']
        rows = []
        for s in D['stages']:
            for k in KS:
                for l in D['labels']:
                    g = lambda q, b: S[(l, s, k, q, b)]['mean']
                    rows.append([s, k, SHOW[l], fnum(g('kernel_norm', 'low'), 3), fnum(g('kernel_norm', 'high'), 3),
                                 fnum(g('kernel_norm', 'high/low'), 3), fnum(g('effective_norm', 'DC'), 3),
                                 fnum(g('effective_norm', 'low'), 3), fnum(g('effective_norm', 'mid'), 3),
                                 fnum(g('effective_norm', 'high'), 3), fnum(g('effective_norm', 'high/low'), 3)])
        L += [table(['단계', 'k', '체크포인트', '|K̂| low', '|K̂| high', '|K̂| high/low', 'ψ|K̂| DC', 'ψ|K̂| low',
                     'ψ|K̂| mid', 'ψ|K̂| high', 'ψ|K̂| high/low'], rows), '']
    return L


def sec1c(M):
    L = ['### 1.3 (c) IN1K 대비 변화량', '',
         '- **IN1K 기준값:** mean|M_IN1K|와 std(M_IN1K)도 함께 적었다. 초기화 randn × 0.02의 기대값은 mean|M| = 0.02·√(2/π) = 0.0160, std = 0.02다.',
         '- **Δψ 대역 평균:** 1.1절 표와 `data/srf_summary.csv`(quantity = delta_psi)에 있다.', '']
    for m, D in M.items():
        rows = []
        for s in D['stages']:
            for k in KS:
                r = D['ref'][(s, k)]
                rows.append([s, k, fnum(r['mean_abs_M_IN1K'], 4), fnum(r['std_M_IN1K'], 4),
                             *[fnum(D['fits'][(l, s, k)]['rel_change'], 3) for l in D['tasks']]])
        L += [f'#### {MODEL_NAME[m]}', '',
              table(['단계', 'k', 'mean|M_IN1K|', 'std(M_IN1K)', *[f'‖ΔM‖/‖M_IN1K‖ {SHOW[l]}' for l in D['tasks']]], rows), '']
    return L


def sec1d(M):
    L = ['### 1.4 (d) weight decay 대조: M_task ≈ a·M_IN1K', '',
         '- **두 배율:** a는 최소제곱 배율이다. a_wd는 0.3절의 decay만 작용했을 때의 잔존 배율로, 단계 안 블록 사이의 범위다.',
         '- **나머지 비:** ‖R‖/‖ΔM‖는 변화 가운데 a·M_IN1K로 설명되지 않는 나머지의 비다.',
         '- **ψ 단위 분해:** residual과 decay_part는 ψ 단위의 대역 평균이다.',
         '- **블록별 값:** `data/fit_decay_by_block.csv`에 있다.', '']
    for m, D in M.items():
        S = D['S']
        wd_by = defaultdict(list)
        for r in D['wd']:
            wd_by[(r['task'], int(r['stage']))].append(r['a_wd_only'])
        rows = []
        for s in D['stages']:
            for k in KS:
                for l in D['tasks']:
                    f = D['fits'][(l, s, k)]
                    aw = wd_by[(l, s)]
                    g = lambda q, b: S[(l, s, k, q, b)]['mean']
                    rows.append([s, k, SHOW[l], fnum(f['a'], 3),
                                 (f'{min(aw):.3f}' + (f'–{max(aw):.3f}' if max(aw) - min(aw) > 5e-4 else '')) if aw else '—',
                                 fnum(f['residual_ratio'], 3), fnum(f['corr'], 3),
                                 fnum(g('decay_part', 'low'), 4, True), fnum(g('decay_part', 'high'), 4, True),
                                 fnum(g('residual', 'low'), 4, True), fnum(g('residual', 'mid'), 4, True),
                                 fnum(g('residual', 'high'), 4, True)])
        L += [f'#### {MODEL_NAME[m]}', '',
              table(['단계', 'k', '체크포인트', 'a', 'a_wd', '‖R‖/‖ΔM‖', 'corr(M_task, M_IN1K)', 'decay_part low',
                     'decay_part high', 'residual low', 'residual mid', 'residual high'], rows), '']
    return L


# ------------------------------------------------------------------------------------------------ (e)
def contrast(M):
    """Per (model, quantity, stage, kernel, band): within-task vs between-task differences of band values."""
    out = []
    for m, D in M.items():
        coco = [l for l in D['tasks'] if l.startswith('COCO')]
        ade = [l for l in D['tasks'] if l.startswith('ADE20K')]
        if not coco or not ade:
            continue
        for q in ('psi', 'residual'):
            for s in D['stages']:
                for k in KS:
                    for bi, b in enumerate(BANDS + ['high/low']):
                        if b == 'high/low':
                            if q != 'psi':
                                continue
                            v = {l: D['S'][(l, s, k, q, b)]['mean'] for l in D['labels'] if (l, s, k, q, b) in D['S']}
                            ch = None
                        else:
                            v = {l: D['S'][(l, s, k, q, b)]['mean'] for l in D['labels'] if (l, s, k, q, b) in D['S']}
                            ch = {l: D['bv'][f'{m}|{l}|s{s}|k{k}|{q}'][:, bi] for l in D['labels']
                                  if f'{m}|{l}|s{s}|k{k}|{q}' in D['bv']}
                        if not all(math.isfinite(v.get(l, float('nan'))) for l in coco + ade):
                            continue
                        mean_d = lambda a, c: abs(v[a] - v[c])
                        rms_d = (lambda a, c: float(np.sqrt(np.mean((ch[a] - ch[c]) ** 2)))) if ch else None
                        r = dict(model=m, quantity=q, stage=s, kernel=k, band=b)
                        for kind, fn in (('mean', mean_d), ('chrms', rms_d)):
                            if fn is None:
                                continue
                            wc = fn(coco[0], coco[1]) if len(coco) == 2 else None
                            wa = fn(ade[0], ade[1]) if len(ade) == 2 else None
                            w = [x for x in (wc, wa) if x is not None]
                            x = [fn(a, c) for a in coco for c in ade]
                            r[f'{kind}_within_COCO'] = float('nan') if wc is None else wc
                            r[f'{kind}_within_ADE'] = float('nan') if wa is None else wa
                            r[f'{kind}_within_max'] = max(w) if w else float('nan')
                            r[f'{kind}_cross_min'] = min(x)
                            r[f'{kind}_cross_max'] = max(x)
                            if kind == 'mean':
                                r['mean_cross_groupmean'] = abs(np.mean([v[l] for l in coco]) - np.mean([v[l] for l in ade]))
                                if q == 'psi' and 'IN1K' in v:
                                    vs = [abs(v[l] - v['IN1K']) for l in coco + ade]
                                    r['mean_vs_IN1K_min'], r['mean_vs_IN1K_max'] = min(vs), max(vs)
                        out.append(r)
    keys = []
    for r in out:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(os.path.join(DATA, 'contrast_e.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in out:
            w.writerow({k: (f'{v:.6g}' if isinstance(v, float) else v) for k, v in r.items()})
    return out


def sec1e(M, C):
    L = ['### 1.5 (e) 같은 과제 안의 차이와 과제 간 차이', '',
         '- **비교 단위:** 단계 × head × 대역마다 각 체크포인트의 대역 값을 비교했다.',
         '- **W (과제 내):** |Mask R-CNN − Cascade|와 |FPN − UperNet| 가운데 큰 값이다. S36 모델은 COCO 체크포인트가 Cascade 하나뿐이라 ADE20K 쌍만 쓴다.',
         '- **X (과제 간):**',
         '  - X_mean = |COCO 쪽 평균 − ADE20K 쪽 평균|.',
         '  - X_min·X_max = COCO 체크포인트와 ADE20K 체크포인트를 하나씩 짝지은 모든 쌍의 |차이| 가운데 최솟값·최댓값.',
         '- **판정:** "X_min > W"는 모든 과제 간 쌍의 차이가 과제 내 차이보다 큰 칸이다.',
         '- **두 수준:**',
         '  - "평균": (블록, 채널) 평균값의 차이.',
         '  - "채널 RMS": 같은 (블록, 채널)끼리의 대역 값 차이의 제곱평균제곱근으로, 평균에서 상쇄되는 채널별 변화까지 잡는다.',
         '- **대상:** ψ 자체와 (d)의 residual(ψ 단위, decay로 설명되지 않는 부분) 두 가지에 대해 계산했다.',
         '- **전체 값:** `data/contrast_e.csv`에 있다.', '',
         '#### 칸 수 요약', '']
    rows = []
    for m, D in M.items():
        for q in ('psi', 'residual'):
            for kind in ('mean', 'chrms'):
                for bset, name in ((('DC', 'low', 'mid', 'high'), 'DC·low·mid·high'), (('high/low',), 'high/low')):
                    cells = [r for r in C if r['model'] == m and r['quantity'] == q and r['band'] in bset
                             and f'{kind}_cross_min' in r]
                    if not cells:
                        continue
                    n = len(cells)
                    a = sum(r[f'{kind}_cross_min'] > r[f'{kind}_within_max'] for r in cells)
                    b = sum(r[f'{kind}_cross_max'] > r[f'{kind}_within_max'] for r in cells)
                    ratio = [r[f'{kind}_cross_min'] / r[f'{kind}_within_max'] for r in cells if r[f'{kind}_within_max'] > 0]
                    rows.append([MODEL_NAME[m], q, '평균' if kind == 'mean' else '채널 RMS', name, n,
                                 f'{a} ({100 * a / n:.0f}%)', f'{b} ({100 * b / n:.0f}%)',
                                 f'{np.median(ratio):.2f}' if ratio else '—'])
    L += [table(['모델', '양', '수준', '대역', '칸 수', 'X_min > W', 'X_max > W', '중앙값 X_min/W'], rows), '']
    L += ['#### ψ 대역 평균: 칸별 값 (W | X_mean, 단위 10⁻³; * = X_min > W)', '']
    for m, D in M.items():
        rows = []
        for s in D['stages']:
            for k in KS:
                row = [s, k]
                for b in BANDS + ['high/low']:
                    r = next((r for r in C if r['model'] == m and r['quantity'] == 'psi' and r['stage'] == s
                              and r['kernel'] == k and r['band'] == b), None)
                    if r is None:
                        row.append('—')
                        continue
                    star = '*' if r['mean_cross_min'] > r['mean_within_max'] else ''
                    row.append(f"{1e3 * r['mean_within_max']:.2f} \\| {1e3 * r['mean_cross_groupmean']:.2f}{star}")
                rows.append(row)
        L += [f'##### {MODEL_NAME[m]}', '', table(['단계', 'k', *[f'{b}: W \\| X' for b in BANDS + ['high/low']]], rows), '']
    L += ['#### residual 대역 평균: 칸별 값 (W | X_mean, 단위 10⁻³; * = X_min > W)', '']
    for m, D in M.items():
        rows = []
        for s in D['stages']:
            for k in KS:
                row = [s, k]
                for b in BANDS:
                    r = next((r for r in C if r['model'] == m and r['quantity'] == 'residual' and r['stage'] == s
                              and r['kernel'] == k and r['band'] == b), None)
                    if r is None:
                        row.append('—')
                        continue
                    star = '*' if r['mean_cross_min'] > r['mean_within_max'] else ''
                    row.append(f"{1e3 * r['mean_within_max']:.2f} \\| {1e3 * r['mean_cross_groupmean']:.2f}{star}")
                rows.append(row)
        L += [f'##### {MODEL_NAME[m]}', '', table(['단계', 'k', *[f'{b}: W \\| X' for b in BANDS]], rows), '']
    return L


def env_lines(M):
    D = next(iter(M.values()))['ins']
    commit = subprocess.run(['git', '-C', REPO, 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    try:
        import mmengine
        mv = mmengine.__version__
    except Exception:
        mv = '-'
    import matplotlib
    return [f"- 저장소 commit `{commit}` (코드는 읽기만 함), Python {D['python']}, torch {D['torch']}, numpy {D['numpy']}, "
            f"mmengine {mv} (체크포인트 메타데이터 읽기와 scheduler 재현), matplotlib {matplotlib.__version__}. CPU만 사용.",
            '- 데이터셋(`~/dataset`의 COCO·ADE20K)은 쓰지 않았다. SRF는 입력과 무관한 파라미터라 체크포인트만으로 계산했다. '
            '추론·학습·평가는 하지 않았다.']


def main():
    M = load()
    C = contrast(M)
    L = ['# 과제별로 미세조정된 SPANetV2의 SRF 마스크 분석', '',
         'ImageNet-1K로 학습한 SPANetV2와, 그 가중치에서 출발해 COCO·ADE20K로 미세조정한 백본의 SRF 마스크(ExSPAM `filter_{1..4}`)를 '
         '대역별로 비교했다. 재현은 `python download_ckpts.py && bash run_all.sh`로 한다.', '']
    L += env_lines(M) + ['']
    L += sec0(M)
    L += sec1_conv(M) + sec1a(M) + sec1b(M) + sec1c(M) + sec1d(M) + sec1e(M, C)
    L += ['## 결론', '', '{{CONCLUSION}}', '']
    L += ['## 파일', '',
          '- `download_ckpts.py`: Release 체크포인트 받기(파일마다 새 디렉터리, 끊기면 이어 받기, Release digest가 있으면 sha256 대조).',
          '- `analyze_srf.py`: 분석 본체. 체크포인트 경로는 `--ckpt LABEL=PATH` 인자로 받는다. `merge`는 모델별 결과를 합친다.',
          '- `plot_srf.py`: 그림. `make_report.py`: 이 보고서와 `data/contrast_e.csv`. `run_all.sh`: 위 전체를 차례로 실행.',
          '- `data/srf_summary.csv`:',
          '  - 열: model, task, ckpt, stage, kernel, quantity, band, mean, p25, p75, n(모은 (블록, 채널) 수).',
          '  - quantity: psi | kernel | kernel_norm | effective | effective_norm | delta_psi | delta_effective_norm | residual | decay_part.',
          '  - band: DC | low | mid | high | high/low.',
          '- `data/srf_radial.npz`:',
          '  - 반경 profile 원자료. key `<model>|<task>|s<stage>|k<kernel>|<quantity>` → 배열 (n, 고리 수).',
          '  - `rho|<model>|s<stage>`는 고리 중심 ρ, `ringw|...`는 고리별 전체 격자 칸 수다.',
          '- `data/fit_decay.csv`:',
          '  - 열: model, task, stage, kernel, a, rel_change, residual_ratio, corr, max_abs_dM, n.',
          '  - 블록별 값은 `data/fit_decay_by_block.csv`에 있다.',
          '- `data/per_model/<모델>/`:',
          '  - `inspect.json`: 0절 원자료.',
          '  - `cfg_<과제>.py`: 체크포인트 안 학습 config.',
          '  - `wd_only_reference.csv`: 블록별 a_wd.',
          '  - `in1k_reference.csv`, `band_values.npz`: (블록, 채널)별 대역 값.',
          '  - `psi_maps.npz`: 2D 지도.',
          '- `figs/`: `psi_radial_*`, `dpsi_radial_*`, `psi_map_*`, `effective_radial_*`, `kernel_radial_*` (모델: S18pure, S36pure, '
          'S18hybrid, S36hybrid). hybrid는 단계 1–2만 SPAM이라 두 단계만 있다.', '']
    text = '\n'.join(L)
    concl_path = os.path.join(HERE, 'conclusion.md')
    concl = open(concl_path).read().strip() if os.path.exists(concl_path) else '(결론 미작성)'
    text = text.replace('{{CONCLUSION}}', concl)
    with open(os.path.join(HERE, 'report.md'), 'w') as f:
        f.write(text)
    print('report.md written', len(text), 'chars')


if __name__ == '__main__':
    main()
