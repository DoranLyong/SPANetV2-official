"""
Re-measure #params and MACs of the COCO detection / ADE20K segmentation models (and, optionally, the
384x384 classification models) with one environment (conda env `spanetv2-dense`) and the convention of
../measure.py (classification, 224x224).

    python measure_dense.py det     # Mask R-CNN 3x, Cascade Mask R-CNN 3x   input 1x3x1280x800
    python measure_dense.py seg     # UperNet 1x3x512x2048, Semantic FPN 1x3x512x512
    python measure_dense.py cls384  # 384x384 classification (optional step 5)
    python measure_dense.py seg --models DFFormer-S18 SPANet-S

CPU, eval mode, randomly initialised weights, torch.rand input (seed 0). Values are MACs.

Per model (full model and, on the same input, its backbone alone):
  (ga) mmengine default: mmengine.analysis.FlopAnalyzer(model, inputs).total()
       = get_model_complexity_info(...)['flops'] used by tools/analysis_tools/get_flops.py
  (a)(b)(c) the handlers of ../measure.py `Convention`, registered unchanged on mmengine's analyser
  (d)  SRF-mask / filter resizing: aten::upsample_bicubic2d, 1 per output element
  total = (ga) + (a) + (b) + (c) + (d)

Detection: all two-stage detectors are counted with the same RoI setting (see TwoStageFixedRoI):
1000 RoIs through every bbox head, the first 100 of them through every mask head.
Segmentation: the auxiliary head is removed before counting, as in semantic_segmentation's get_flops.py.
"""
import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
FB = os.path.dirname(HERE)
ROOT = os.path.dirname(FB)
sys.path.insert(0, FB)
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import measure as M  # noqa: E402  (../measure.py: Convention, aggregate, fft_checks, unsupported_by_module, quiet)
import zoo  # noqa: E402

OUT = os.path.join(HERE, 'results')
TP = zoo.TP
DET_CFG = os.path.join(ROOT, 'object_detection', 'local_configs', 'SPANetV2')
SEG_CFG = os.path.join(ROOT, 'semantic_segmentation', 'local_configs')
DFF_CFG = os.path.join(TP, 'dfformer', 'segmentation', 'configs', 'fpn')
C = os.path.join(HERE, 'configs')

SETTINGS = {
    'mask_rcnn': dict(task='det', hw=(1280, 800), title='COCO Mask R-CNN 3×'),
    'cascade':   dict(task='det', hw=(1280, 800), title='COCO Cascade Mask R-CNN 3×'),
    'upernet':   dict(task='seg', hw=(512, 2048), title='ADE20K UperNet'),
    'fpn':       dict(task='seg', hw=(512, 512),  title='ADE20K Semantic FPN'),
    'cls384':    dict(task='cls384', hw=(384, 384), title='ImageNet-1K 384×384 (분류)'),
}
NUM_ROIS, NUM_MASK_ROIS = 1000, 100


def _d(p):
    return os.path.join(DET_CFG, p)


def _s(p):
    return os.path.join(SEG_CFG, p)


# (setting, name, family, size group, pure/hybrid, paper (params M or None, MACs G), source, config / zoo name)
# source: 'repo' = config from the model's repository, 'constructed' = written for this measurement (configs/),
#         'zoo' = classification model built by ../zoo.py
MODELS = [
    # ---- COCO Mask R-CNN 3x
    ('mask_rcnn', 'ConvFormer-S18',      'ConvFormer',      'S18', 'pure',   (None, 251), 'constructed', os.path.join(C, 'det/mask-rcnn_convformer_s18_3x.py')),
    ('mask_rcnn', 'SPANetV2-S18-pure',   'SPANetV2-pure',   'S18', 'pure',   (None, 255), 'repo',        _d('mask-rcnn_spanetv2_s18_pure_3x.py')),
    ('mask_rcnn', 'CAFormer-S18',        'CAFormer',        'S18', 'hybrid', (None, 254), 'constructed', os.path.join(C, 'det/mask-rcnn_caformer_s18_3x.py')),
    ('mask_rcnn', 'SPANetV2-S18-hybrid', 'SPANetV2-hybrid', 'S18', 'hybrid', (None, 251), 'repo',        _d('mask-rcnn_spanetv2_s18_hybrid_3x.py')),
    # ---- COCO Cascade Mask R-CNN 3x
    ('cascade', 'ConvFormer-S18',      'ConvFormer',      'S18', 'pure',   (None, 729), 'constructed', os.path.join(C, 'det/cascade-mask-rcnn_convformer_s18_fpn-3x_coco.py')),
    ('cascade', 'SPANetV2-S18-pure',   'SPANetV2-pure',   'S18', 'pure',   (None, 734), 'repo',        _d('cascade-mask-rcnn_spanetv2_s18_pure_fpn-3x_coco.py')),
    ('cascade', 'CAFormer-S18',        'CAFormer',        'S18', 'hybrid', (None, 733), 'constructed', os.path.join(C, 'det/cascade-mask-rcnn_caformer_s18_fpn-3x_coco.py')),
    ('cascade', 'SPANetV2-S18-hybrid', 'SPANetV2-hybrid', 'S18', 'hybrid', (None, 729), 'repo',        _d('cascade-mask-rcnn_spanetv2_s18_hybrid_fpn-3x_coco.py')),
    ('cascade', 'ConvFormer-S36',      'ConvFormer',      'S36', 'pure',   (None, 805), 'constructed', os.path.join(C, 'det/cascade-mask-rcnn_convformer_s36_fpn-3x_coco.py')),
    ('cascade', 'SPANetV2-S36-pure',   'SPANetV2-pure',   'S36', 'pure',   (None, 814), 'constructed', os.path.join(C, 'det/cascade-mask-rcnn_spanetv2_s36_pure_fpn-3x_coco.py')),
    ('cascade', 'CAFormer-S36',        'CAFormer',        'S36', 'hybrid', (None, 811), 'constructed', os.path.join(C, 'det/cascade-mask-rcnn_caformer_s36_fpn-3x_coco.py')),
    ('cascade', 'SPANetV2-S36-hybrid', 'SPANetV2-hybrid', 'S36', 'hybrid', (None, 806), 'constructed', os.path.join(C, 'det/cascade-mask-rcnn_spanetv2_s36_hybrid_fpn-3x_coco.py')),
    # ---- ADE20K UperNet
    ('upernet', 'ConvFormer-S18',      'ConvFormer',      'S18', 'pure',   (54, 925),  'constructed', os.path.join(C, 'seg/upernet_convformer_s18_ade20k-160k.py')),
    ('upernet', 'SPANetV2-S18-pure',   'SPANetV2-pure',   'S18', 'pure',   (60, 930),  'repo',        _s('upernet/SPANetV2/upernet_spanet_s18_pure_ade20k-160k.py')),
    ('upernet', 'CAFormer-S18',        'CAFormer',        'S18', 'hybrid', (54, 1024), 'constructed', os.path.join(C, 'seg/upernet_caformer_s18_ade20k-160k.py')),
    ('upernet', 'SPANetV2-S18-hybrid', 'SPANetV2-hybrid', 'S18', 'hybrid', (58, 925),  'repo',        _s('upernet/SPANetV2/upernet_spanet_s18_hybrid_ade20k-160k.py')),
    ('upernet', 'ConvFormer-S36',      'ConvFormer',      'S36', 'pure',   (67, 1003), 'constructed', os.path.join(C, 'seg/upernet_convformer_s36_ade20k-160k.py')),
    ('upernet', 'SPANetV2-S36-pure',   'SPANetV2-pure',   'S36', 'pure',   (75, 1012), 'repo',        _s('upernet/SPANetV2/upernet_spanet_s36_pure_ade20k-160k.py')),
    ('upernet', 'CAFormer-S36',        'CAFormer',        'S36', 'hybrid', (67, 1197), 'constructed', os.path.join(C, 'seg/upernet_caformer_s36_ade20k-160k.py')),
    ('upernet', 'SPANetV2-S36-hybrid', 'SPANetV2-hybrid', 'S36', 'hybrid', (71, 1004), 'repo',        _s('upernet/SPANetV2/upernet_spanet_s36_hybrid_ade20k-160k.py')),
    # ---- ADE20K Semantic FPN
    ('fpn', 'SPANet-S',            'SPANet',          'S18', 'pure',   (32, 46), 'constructed', os.path.join(C, 'seg/fpn_spanet_s_ade20k-40k.py')),
    ('fpn', 'DFFormer-S18',        'DFFormer',        'S18', 'pure',   (35, 41), 'repo',        os.path.join(DFF_CFG, 'dfformer_s18_fpn.py')),
    ('fpn', 'SPANetV2-S18-pure',   'SPANetV2-pure',   'S18', 'pure',   (33, 43), 'repo',        _s('fpn/SPANetV2/fpn_spanet_s18_pure_ade20k-40k.py')),
    ('fpn', 'CDFFormer-S18',       'CDFFormer',       'S18', 'hybrid', (35, 42), 'repo',        os.path.join(DFF_CFG, 'cdfformer_s18_fpn.py')),
    ('fpn', 'SPANetV2-S18-hybrid', 'SPANetV2-hybrid', 'S18', 'hybrid', (31, 48), 'repo',        _s('fpn/SPANetV2/fpn_spanet_s18_hybrid_ade20k-40k.py')),
    ('fpn', 'SPANet-M',            'SPANet',          'S36', 'pure',   (45, 57), 'constructed', os.path.join(C, 'seg/fpn_spanet_m_ade20k-40k.py')),
    ('fpn', 'DFFormer-S36',        'DFFormer',        'S36', 'pure',   (50, 60), 'repo',        os.path.join(DFF_CFG, 'dfformer_s36_fpn.py')),
    ('fpn', 'SPANetV2-S36-pure',   'SPANetV2-pure',   'S36', 'pure',   (48, 64), 'repo',        _s('fpn/SPANetV2/fpn_spanet_s36_pure_ade20k-40k.py')),
    ('fpn', 'CDFFormer-S36',       'CDFFormer',       'S36', 'hybrid', (50, 61), 'repo',        os.path.join(DFF_CFG, 'cdfformer_s36_fpn.py')),
    ('fpn', 'SPANetV2-S36-hybrid', 'SPANetV2-hybrid', 'S36', 'hybrid', (45, 74), 'repo',        _s('fpn/SPANetV2/fpn_spanet_s36_hybrid_ade20k-40k.py')),
    # ---- 384x384 classification (optional)
    ('cls384', 'ConvFormer-S18',      'ConvFormer',      'S18', 'pure',   (None, 11.6), 'zoo', 'ConvFormer-S18'),
    ('cls384', 'SPANetV2-S18-pure',   'SPANetV2-pure',   'S18', 'pure',   (None, 12.3), 'zoo', 'SPANetV2-S18-pure'),
    ('cls384', 'CAFormer-S18',        'CAFormer',        'S18', 'hybrid', (None, 13.4), 'zoo', 'CAFormer-S18'),
    ('cls384', 'SPANetV2-S18-hybrid', 'SPANetV2-hybrid', 'S18', 'hybrid', (None, 13.6), 'zoo', 'SPANetV2-S18-hybrid'),
    ('cls384', 'ConvFormer-S36',      'ConvFormer',      'S36', 'pure',   (None, 22.4), 'zoo', 'ConvFormer-S36'),
    ('cls384', 'SPANetV2-S36-pure',   'SPANetV2-pure',   'S36', 'pure',   (None, 23.9), 'zoo', 'SPANetV2-S36-pure'),
    ('cls384', 'CAFormer-S36',        'CAFormer',        'S36', 'hybrid', (None, 26.0), 'zoo', 'CAFormer-S36'),
    ('cls384', 'SPANetV2-S36-hybrid', 'SPANetV2-hybrid', 'S36', 'hybrid', (None, 26.6), 'zoo', 'SPANetV2-S36-hybrid'),
]


# ---------------------------------------------------------------------------------------------
# Convention: ../measure.py handlers unchanged, plus (d) and the ops that only occur in the dense models.
INTERP_OPS = {'aten::upsample_bicubic2d'}       # (d)
# Ops that do not occur in the 224x224 classification models. Each is listed in the README.
EXTRA_ELEMENTWISE = {'aten::log2'}              # -> (c), 1 per output element (same rule as measure.ELEMENTWISE_OPS)
EXTRA_ZERO = {                                  # op -> note, counted 0
    'aten::feature_dropout': 'nn.Dropout2d in eval mode (identity; aten::dropout is on the ignore list)',
}
NOT_COUNTED = {                                 # op -> note, counted 0 and logged (not element-wise arithmetic)
    'prim::PythonOp.RoIAlignFunction': 'mmcv RoIAlign (bilinear sampling of RoI features); not counted',
}


class DenseConvention(M.Convention):
    def handles(self):
        h = super().handles()
        for op in INTERP_OPS:
            h[op] = self._interp(op)
        for op in EXTRA_ELEMENTWISE:
            h[op] = self._elementwise(op)
        for op, note in EXTRA_ZERO.items():
            h[op] = self._fixed(op, 'zero', note)
        for op, note in NOT_COUNTED.items():
            h[op] = self._fixed(op, 'notcounted', note)
        h['aten::max_pool2d'] = self._maxpool('aten::max_pool2d')
        return h

    def _maxpool(self, op):
        """FPN's extra output level (mmdet FPN, num_outs=5): F.max_pool2d(x, 1, stride=2). A 1x1 window has no
        arithmetic, so it is 0 (subsampling). Any other window size stops the measurement."""
        def handle(inputs, outputs):
            out = outputs[0]
            k = M.ivalue(inputs[1])
            k = list(k) if isinstance(k, (list, tuple)) else [k]
            if math.prod(k) != 1:
                raise RuntimeError(f'{op} with kernel {k} at {M.scope_of(out)}: no rule in the convention')
            shape = M.get_shape(out)
            self.records.append(dict(op=op, cat='zero', module=M.scope_of(out), count=0,
                                     note=f'kernel {k}, stride {M.ivalue(inputs[2])}: subsampling, no arithmetic',
                                     out_shape=shape, out_dtype=M.dtype_of(out), numel=math.prod(shape)))
            return 0
        return handle

    def _interp(self, op):
        def handle(inputs, outputs):
            x, out = inputs[0], outputs[0]
            shape = M.get_shape(out)
            n = math.prod(shape)
            args = []
            for v in inputs[1:]:
                try:
                    args.append(M.ivalue(v))
                except Exception:
                    args.append('?')
            self.records.append(dict(op=op, cat='d', module=M.scope_of(out), count=n,
                                     in_shape=M.get_shape(x), in_dtype=M.dtype_of(x),
                                     out_shape=shape, out_dtype=M.dtype_of(out), args=args))
            return n
        return handle


# ---------------------------------------------------------------------------------------------
class TwoStageFixedRoI(nn.Module):
    """Two-stage detector forward with a fixed RoI count, the same for every model.

    Same structure as mmdet 2.x `TwoStageDetector.forward_dummy` + `{Standard,Cascade}RoIHead.forward_dummy`
    (backbone -> neck -> RPN head convs -> every bbox head on all RoIs -> every mask head on the first 100 RoIs),
    written with the mmdet 3.x RoI-head methods `_bbox_forward` / `_mask_forward` of the built model.
    The submodules are the detector's own modules under their own names (backbone, neck, rpn_head, roi_head).
    """

    def __init__(self, det, num_mask_rois=NUM_MASK_ROIS):
        super().__init__()
        self.backbone, self.neck, self.rpn_head, self.roi_head = det.backbone, det.neck, det.rpn_head, det.roi_head
        self.num_mask_rois = num_mask_rois
        self.cascade = hasattr(det.roi_head, 'num_stages')

    def forward(self, img, rois):
        x = self.neck(self.backbone(img))
        cls_scores, bbox_preds = self.rpn_head(x)
        outs = list(cls_scores) + list(bbox_preds)
        mask_rois = rois[:self.num_mask_rois]
        if self.cascade:
            for i in range(self.roi_head.num_stages):
                r = self.roi_head._bbox_forward(i, x, rois)
                outs += [r['cls_score'], r['bbox_pred']]
            for i in range(self.roi_head.num_stages):
                outs.append(self.roi_head._mask_forward(i, x, mask_rois)['mask_preds'])
        else:
            r = self.roi_head._bbox_forward(x, rois)
            outs += [r['cls_score'], r['bbox_pred']]
            outs.append(self.roi_head._mask_forward(x, mask_rois)['mask_preds'])
        return tuple(outs)


def make_rois(n, h, w, seed=0):
    """n fixed boxes inside the image, (batch_idx, x1, y1, x2, y2). Box values do not change any count."""
    g = torch.Generator().manual_seed(seed)
    x1 = torch.rand(n, generator=g) * (w - 32)
    y1 = torch.rand(n, generator=g) * (h - 32)
    x2 = x1 + 16 + torch.rand(n, generator=g) * (w - x1 - 16)
    y2 = y1 + 16 + torch.rand(n, generator=g) * (h - y1 - 16)
    return torch.stack([torch.zeros(n), x1, y1, x2, y2], dim=1)


# ---------------------------------------------------------------------------------------------
def setup_task(task):
    """Put the right mmdet / mmseg copy on sys.path and register the extra backbones."""
    if task == 'det':
        sys.path.insert(0, os.path.join(ROOT, 'object_detection'))
        import mmdet.models  # noqa: F401  (registers spanetv2_* backbones)
        from mmdet.registry import MODELS as REG
        from mmengine.registry import init_default_scope
        init_default_scope('mmdet')
        import backbones
        backbones.register_metaformer(REG)
        return REG
    if task == 'seg':
        sys.path.insert(0, os.path.join(ROOT, 'semantic_segmentation'))
        import mmseg.models  # noqa: F401
        from mmseg.registry import MODELS as REG
        from mmengine.registry import init_default_scope
        init_default_scope('mmseg')
        import backbones
        backbones.register_metaformer(REG)
        with backbones.captured_stdout():
            backbones.register_spanet_v1()
        backbones.register_dfformer_seg()
        return REG
    return None


def build_dense(reg, cfg_path, cfg_options=None):
    from mmengine.config import Config
    from mmengine.model import revert_sync_batchnorm
    import backbones
    cfg = Config.fromfile(cfg_path)
    overrides = []
    if cfg_options:                                 # only for the configuration check (config_check_dense.py)
        cfg.merge_from_dict(cfg_options)
        overrides += [f'{k}={v}' for k, v in cfg_options.items()]
    bb = cfg.model.backbone
    if bb.get('init_cfg') is not None:
        bb.init_cfg = None
        overrides.append('model.backbone.init_cfg=None (no checkpoint is loaded)')
    torch.manual_seed(0)
    with backbones.captured_stdout() as buf:
        model = reg.build(cfg.model)
    model = revert_sync_batchnorm(model)
    model.eval()
    return model, cfg, overrides, buf.getvalue()


def nparams(m):
    return sum(p.numel() for p in m.parameters()) if m is not None else 0


def unused_modules(fa, target):
    """Modules that neither they nor any submodule were called in the trace (top-most only), with #params."""
    uncalled = set(fa.uncalled_modules())
    names = dict(target.named_modules())
    dead = set()
    for n in uncalled:
        pre = n + '.'
        if all((d in uncalled) for d in names if d.startswith(pre)):
            dead.add(n)
    top = [n for n in dead if not any(n.startswith(p + '.') for p in dead if p != n)]
    return sorted([dict(module=n, cls=type(names[n]).__name__, params=nparams(names[n])) for n in top if n in names],
                  key=lambda r: r['module'])


def analyse(target, inputs):
    """(ga) mmengine default + the convention, for one target module."""
    from mmengine.analysis import FlopAnalyzer
    res = {}
    fa = M.quiet(FlopAnalyzer(target, inputs))
    res['ga'] = fa.total()
    res['ga_by_operator'] = dict(fa.by_operator())
    res['ga_by_module_top'] = {k: v for k, v in fa.by_module().items() if k and '.' not in k}
    res['unsupported_ops_raw'] = repr(fa.unsupported_ops())
    res['unsupported_ops'] = dict(fa.unsupported_ops())
    res['unsupported_by_module'] = M.unsupported_by_module(fa, target)
    res['unused_modules'] = unused_modules(fa, target)

    conv = DenseConvention()
    with torch.no_grad():
        fc = M.quiet(FlopAnalyzer(target, inputs))
        fc.set_op_handle(**conv.handles())
        total_c = fc.total()
        left = dict(fc.unsupported_ops())
        res['total_by_module_top'] = {k: v for k, v in fc.by_module().items() if k and '.' not in k}
    if left:
        raise RuntimeError(f'ops without a handler in the convention: {left}')
    for cat in ('a', 'b', 'c', 'd'):
        res[cat] = sum(r['count'] for r in conv.records if r['cat'] == cat)
    res['total'] = res['ga'] + res['a'] + res['b'] + res['c'] + res['d']
    assert math.isclose(total_c, res['total'], rel_tol=1e-12), (total_c, res['total'])
    res['convention_records'] = M.aggregate(conv.records)
    res['fft_checks'] = M.fft_checks(conv.records)
    seen, ic = set(), []
    for r in conv.records:
        if r['cat'] == 'd' and r['op'] not in seen:
            seen.add(r['op'])
            ic.append({k: r[k] for k in ('op', 'module', 'in_shape', 'in_dtype', 'out_shape', 'out_dtype', 'args', 'count')})
    res['interp_checks'] = ic
    return res


def measure_one(entry, reg, cfg_options=None):
    setting, name, family, group, kind, paper, source, ref = entry
    st = SETTINGS[setting]
    H, W = st['hw']
    t0 = time.time()
    r = dict(setting=setting, name=name, family=family, group=group, kind=kind, paper_params_M=paper[0],
             paper_macs_G=paper[1], source=source, input=[1, 3, H, W])
    torch.manual_seed(0)
    x = torch.rand(1, 3, H, W)
    if st['task'] == 'cls384':
        model, ctor = zoo.build(ref)
        r.update(config=None, ctor_stdout=ctor, overrides=[])
        target, inputs, backbone = model, (x,), None
    else:
        model, cfg, overrides, ctor = build_dense(reg, ref, cfg_options)
        r.update(config=os.path.relpath(ref, HERE), ctor_stdout=ctor, overrides=overrides,
                 backbone_type=cfg.model.backbone.type, backbone_cfg=str(cfg.model.backbone))
        if st['task'] == 'det':
            rois = make_rois(NUM_ROIS, H, W)
            target, inputs = TwoStageFixedRoI(model), (x, rois)
            r.update(num_rois=NUM_ROIS, num_mask_rois=NUM_MASK_ROIS, detector=type(model).__name__,
                     roi_head=type(model.roi_head).__name__,
                     num_stages=getattr(model.roi_head, 'num_stages', 1))
            r['params_detector'] = nparams(model)
        else:
            r['params_auxiliary_head'] = nparams(getattr(model, 'auxiliary_head', None))
            r['params_with_auxiliary_head'] = nparams(model)
            if hasattr(model, 'auxiliary_head'):
                model.auxiliary_head = None         # as semantic_segmentation/tools/analysis_tools/get_flops.py
            target, inputs = model, (x,)
            r['decode_head'] = type(model.decode_head).__name__
        backbone = model.backbone
    r['params'] = nparams(target)
    r.update(analyse(target, inputs))
    if backbone is not None:
        r['params_backbone'] = nparams(backbone)
        r['backbone'] = analyse(backbone, (x,))
    r['seconds'] = round(time.time() - t0, 1)
    return r


# ---------------------------------------------------------------------------------------------
def git(repo, *args):
    return subprocess.run(['git', '-C', repo, *args], capture_output=True, text=True).stdout.strip()


def environment(task):
    import fvcore
    import mmcv
    import mmengine
    import timm
    env = dict(date=time.strftime('%Y-%m-%d %H:%M:%S %Z'), python=sys.version.split()[0], torch=torch.__version__,
               mmengine=mmengine.__version__, mmcv=mmcv.__version__, fvcore=getattr(fvcore, '__version__', '?'),
               timm=timm.__version__, numpy=np.__version__, device='cpu', torch_threads=torch.get_num_threads(),
               platform=platform.platform(),
               cpu=subprocess.run("lscpu | sed -n 's/^Model name: *//p'", shell=True, capture_output=True,
                                  text=True).stdout.strip(),
               python_executable=sys.executable)
    if task == 'det':
        import mmdet
        env['mmdet'] = f'{mmdet.__version__} ({os.path.relpath(os.path.dirname(mmdet.__file__), HERE)})'
    if task == 'seg':
        import mmseg
        env['mmseg'] = f'{mmseg.__version__} ({os.path.relpath(os.path.dirname(mmseg.__file__), HERE)})'
    repos = {}
    for key, root in [('SPANetV2-official', ROOT), ('metaformer', os.path.join(TP, 'metaformer')),
                      ('dfformer', os.path.join(TP, 'dfformer')), ('SPANet-official', os.path.join(TP, 'SPANet-official'))]:
        repos[key] = dict(root=os.path.relpath(root, HERE), commit=git(root, 'rev-parse', 'HEAD'),
                          remote=git(root, 'remote', 'get-url', 'origin'),
                          status=git(root, 'status', '--short', '--untracked-files=no') or 'clean')
    env['repos'] = repos
    return env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('task', choices=['det', 'seg', 'cls384'])
    ap.add_argument('--models', nargs='*', default=None)
    ap.add_argument('--settings', nargs='*', default=None)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    reg = setup_task(args.task)
    env = environment(args.task)
    print(json.dumps(env, indent=1), flush=True)
    with open(os.path.join(OUT, f'env_{args.task}.json'), 'w') as f:
        json.dump(env, f, indent=1)

    path = os.path.join(OUT, f'{args.task}.json')
    results = json.load(open(path)) if ((args.models or args.settings) and os.path.exists(path)) else {}
    for entry in MODELS:
        setting, name = entry[0], entry[1]
        if SETTINGS[setting]['task'] != args.task:
            continue
        if args.settings and setting not in args.settings:
            continue
        if args.models and name not in args.models:
            continue
        r = measure_one(entry, reg)
        results[f'{setting}/{name}'] = r
        bb = r.get('backbone')
        print(f"{setting:9s} {name:20s} params {r['params'] / 1e6:8.3f}M | paper {r['paper_macs_G']} | "
              f"ga {r['ga'] / 1e9:9.3f} a {r['a'] / 1e9:.3f} b {r['b'] / 1e9:.3f} c {r['c'] / 1e9:.3f} "
              f"d {r['d'] / 1e9:.3f} | total {r['total'] / 1e9:9.3f}"
              + (f" | backbone {bb['total'] / 1e9:8.3f}" if bb else '') + f" | {r['seconds']}s", flush=True)
        with open(path, 'w') as f:
            json.dump(results, f, indent=1, default=str)


if __name__ == '__main__':
    main()
