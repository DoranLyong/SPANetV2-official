#!/usr/bin/env python
"""D3: single-scale COCO val2017 inference (mmdet 3.x copy of SPANetV2-official): box / mask AP and result files.

    python det_eval.py --model cascade_spanetv2_s18_pure [--weight PATH] [--limit N]

Writes cache/preds/det/<model>/results.{bbox,segm}.json (COCO results, RLE masks) and data/raw/det_<model>.json.
Boundary AP is computed from results.segm.json by boundary_ap.py.

Models
  SPANetV2         config = the training config saved in the checkpoint (meta['cfg']).
  ConvNeXt-T/S,    official backbone files (ConvNeXt object_detection/mmdet/models/backbones/convnext.py,
  Swin-T/S         Swin-Transformer-Object-Detection mmdet/models/backbones/swin_transformer.py), imported unchanged
                   with the same temporary sys.modules stand-ins as seg_eval.py (`mmcv_custom`, `mmdet.models.builder`,
                   `mmdet.utils.get_root_logger`). Detector = the SPANetV2 S18 checkpoint config of the same framework
                   (Mask R-CNN or Cascade Mask R-CNN 4conv1fc-GIoU-SyncBN heads, test 1333x800, rpn 1000, rcnn 100,
                   score 0.05, nms 0.5, mask 0.5 — the same values as the official configs) with the official backbone
                   arguments and neck in_channels [96, 192, 384, 768].
"""
import argparse
import copy
import importlib.util
import json
import os
import sys
import tempfile
import time
import types

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
TP = os.path.join(HERE, 'third_party')
COCO = os.path.join(HERE, 'cache', 'datasets', 'coco')
PREDS = os.path.join(HERE, 'cache', 'preds', 'det')
RAW = os.path.join(HERE, 'data', 'raw')

CNX = dict(type='OfficialConvNeXt', in_chans=3, dims=[96, 192, 384, 768], layer_scale_init_value=1.0,
           out_indices=[0, 1, 2, 3])
SWIN = dict(type='OfficialSwinTransformer', embed_dim=96, num_heads=[3, 6, 12, 24], window_size=7, mlp_ratio=4.,
            qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., ape=False, patch_norm=True,
            out_indices=(0, 1, 2, 3), use_checkpoint=False)
REF_SRC = {'convnext': 'facebookresearch/ConvNeXt object_detection README', 'swin': 'Swin-Transformer-Object-Detection README',
           'spanetv2': 'SPANetV2-official README'}

# key: (name, framework, pair, backbone cfg or None (= checkpoint cfg), template key, reference (box AP, mask AP))
DET_MODELS = {
    'mask_rcnn_spanetv2_s18_pure': ('SPANetV2-S18-pure', 'mask_rcnn', 'S18', None, None, (48.0, 42.9)),
    'mask_rcnn_spanetv2_s18_hybrid': ('SPANetV2-S18-hybrid', 'mask_rcnn', 'S18', None, None, (49.6, 44.3)),
    'cascade_spanetv2_s18_pure': ('SPANetV2-S18-pure', 'cascade', 'S18', None, None, (51.6, 44.7)),
    'cascade_spanetv2_s18_hybrid': ('SPANetV2-S18-hybrid', 'cascade', 'S18', None, None, (52.8, 45.7)),
    'cascade_spanetv2_s36_pure': ('SPANetV2-S36-pure', 'cascade', 'S36', None, None, (53.1, 45.9)),
    'cascade_spanetv2_s36_hybrid': ('SPANetV2-S36-hybrid', 'cascade', 'S36', None, None, (53.8, 46.4)),
    'mask_rcnn_convnext_tiny': ('ConvNeXt-T', 'mask_rcnn', 'S18', dict(CNX, depths=[3, 3, 9, 3], drop_path_rate=0.4),
                                'mask_rcnn_spanetv2_s18_pure', (46.2, 41.7)),
    'cascade_convnext_tiny': ('ConvNeXt-T', 'cascade', 'S18', dict(CNX, depths=[3, 3, 9, 3], drop_path_rate=0.4),
                              'cascade_spanetv2_s18_pure', (50.4, 43.7)),
    'cascade_convnext_small': ('ConvNeXt-S', 'cascade', 'S36', dict(CNX, depths=[3, 3, 27, 3], drop_path_rate=0.6),
                               'cascade_spanetv2_s18_pure', (51.9, 45.0)),
    'mask_rcnn_swin_tiny': ('Swin-T', 'mask_rcnn', 'S18', dict(SWIN, depths=[2, 2, 6, 2], drop_path_rate=0.2),
                            'mask_rcnn_spanetv2_s18_pure', (46.0, 41.6)),
    'cascade_swin_tiny': ('Swin-T', 'cascade', 'S18', dict(SWIN, depths=[2, 2, 6, 2], drop_path_rate=0.2),
                          'cascade_spanetv2_s18_pure', (50.4, 43.7)),
    'cascade_swin_small': ('Swin-S', 'cascade', 'S36', dict(SWIN, depths=[2, 2, 18, 2], drop_path_rate=0.2),
                           'cascade_spanetv2_s18_pure', (51.9, 45.0)),
}


def setup_mmdet():
    sys.path.insert(0, os.path.join(REPO, 'object_detection'))
    import mmdet.datasets  # noqa: F401
    import mmdet.evaluation  # noqa: F401
    import mmdet.models  # noqa: F401
    from mmengine.registry import init_default_scope
    init_default_scope('mmdet')


class _OfficialRegistry:
    def register_module(self, name=None, force=False, module=None):
        from mmdet.registry import MODELS

        def deco(cls):
            MODELS.register_module(name=f'Official{name or cls.__name__}', module=cls, force=True)
            return cls
        return deco(module) if module is not None else deco


def load_official(path, modname):
    from mmengine.logging import MMLogger
    from mmengine.runner import load_checkpoint
    import mmdet.utils
    stubs = {'mmcv_custom': types.SimpleNamespace(load_checkpoint=load_checkpoint),
             'mmdet.models.builder': types.SimpleNamespace(BACKBONES=_OfficialRegistry())}
    saved = {k: sys.modules.get(k) for k in stubs}
    had = hasattr(mmdet.utils, 'get_root_logger')
    sys.modules.update(stubs)
    if not had:
        mmdet.utils.get_root_logger = lambda *a, **k: MMLogger.get_current_instance()
    try:
        spec = importlib.util.spec_from_file_location(modname, path)
        mod = importlib.util.module_from_spec(spec)
        mod.__package__ = 'mmdet.models.backbones'          # for `from ..builder import BACKBONES`
        sys.modules[modname] = mod
        old, sys.dont_write_bytecode = sys.dont_write_bytecode, True
        try:
            spec.loader.exec_module(mod)
        finally:
            sys.dont_write_bytecode = old
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
        if not had:
            del mmdet.utils.get_root_logger
    return mod


def load_weights(path):
    info = {}
    try:
        ck = torch.load(path, map_location='cpu', weights_only=True)
        info['weights_only'] = True
    except Exception as e:
        info['weights_only'] = False
        info['weights_only_true_error'] = f'{type(e).__name__}: ' + ' '.join(str(e).split())[:300]
        ck = torch.load(path, map_location='cpu', weights_only=False)
    return ck, info


def ckpt_config(ck):
    from mmengine.config import Config
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 'cfg.py')
        with open(p, 'w') as f:
            f.write(ck['meta']['cfg'])
        return Config.fromfile(p, import_custom_modules=False)


def weight_path(key):
    """data/weights_manifest.json entry, or the download location while download_weights.py is still running."""
    with open(os.path.join(HERE, 'data', 'weights_manifest.json')) as f:
        man = json.load(f)
    if key in man:
        return os.path.join(HERE, man[key]['path'])
    spec = importlib.util.spec_from_file_location('_dw', os.path.join(HERE, 'download_weights.py'))
    dw = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dw)
    for g in ('seg', 'det'):
        if key in dw.WEIGHTS[g]:
            return dw.path_for(g, dw.WEIGHTS[g][key])
    raise KeyError(key)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=list(DET_MODELS))
    ap.add_argument('--weight', default=None)
    ap.add_argument('--template-weight', default=None)
    ap.add_argument('--limit', type=int, default=None)
    args = ap.parse_args()
    setup_mmdet()
    from mmengine.dataset import pseudo_collate
    from mmengine.model import revert_sync_batchnorm
    from mmdet.registry import DATASETS, METRICS, MODELS
    load_official(os.path.join(TP, 'ConvNeXt', 'object_detection', 'mmdet', 'models', 'backbones', 'convnext.py'),
                  'mmdet.models.backbones._fr_convnext_det')
    load_official(os.path.join(TP, 'Swin-Transformer-Object-Detection', 'mmdet', 'models', 'backbones',
                               'swin_transformer.py'), 'mmdet.models.backbones._fr_swin_det')

    name, fw, pair, bb, template, ref = DET_MODELS[args.model]
    wpath = args.weight or weight_path(args.model)
    ck, info = load_weights(wpath)
    if bb is None:
        cfg = ckpt_config(ck)
        cfg_src = "checkpoint meta['cfg']"
    else:
        tck, _ = load_weights(args.template_weight or weight_path(template))
        cfg = ckpt_config(tck)
        del tck
        cfg.model.backbone = bb
        cfg.model.neck.in_channels = [96, 192, 384, 768]
        cfg_src = f"checkpoint meta['cfg'] of {template} with the official backbone (see det_eval.py)"
    mcfg = copy.deepcopy(cfg.model)
    mcfg['backbone'].pop('init_cfg', None)
    sd = ck['state_dict'] if 'state_dict' in ck else ck
    torch.manual_seed(0)
    model = MODELS.build(mcfg)
    res = model.load_state_dict(sd, strict=False)
    info.update(missing=list(res.missing_keys), unexpected=list(res.unexpected_keys))
    meta = ck.get('meta', {}) if isinstance(ck, dict) else {}
    info['meta'] = {k: (v if isinstance(v, (int, float, str)) else str(v))
                    for k, v in meta.items() if k in ('epoch', 'iter', 'mmdet_version', 'mmcv_version', 'time')}
    del ck, sd
    model = revert_sync_batchnorm(model).cuda().eval()

    dcfg = copy.deepcopy(cfg.test_dataloader.dataset)
    dcfg.update(data_root=COCO + '/', ann_file='annotations/instances_val2017.json',
                data_prefix=dict(img='val2017/'), test_mode=True)
    ds = DATASETS.build(dcfg)
    out_dir = os.path.join(PREDS, args.model)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(RAW, exist_ok=True)
    metric = METRICS.build(dict(type='CocoMetric', ann_file=os.path.join(COCO, 'annotations', 'instances_val2017.json'),
                                metric=['bbox', 'segm'], format_only=False,
                                outfile_prefix=os.path.join(out_dir, 'results')))
    metric.dataset_meta = ds.metainfo
    n = len(ds) if args.limit is None else args.limit
    t0 = time.time()
    with torch.no_grad():
        for i in range(n):
            batch = pseudo_collate([ds[i]])
            outs = model.test_step(batch)
            metric.process(batch, [o.to_dict() if hasattr(o, 'to_dict') else o for o in outs])
            if i % 500 == 0:
                print(f'{i}/{n} {time.time() - t0:.0f}s', flush=True)
    if args.limit is not None:                          # smoke test: evaluate only the processed images
        metric.img_ids = [r[1]['img_id'] for r in metric.results]
    m = metric.compute_metrics(metric.results)
    result = dict(model=args.model, name=name, framework=fw, pair=pair, n_images=n,
                  metrics={k: float(v) for k, v in m.items() if isinstance(v, (int, float))},
                  reference=dict(box_AP=ref[0], mask_AP=ref[1],
                                 src=REF_SRC['spanetv2' if bb is None else ('convnext' if 'convnext' in args.model else 'swin')]),
                  config_source=cfg_src, load_info=info, weight=os.path.relpath(wpath, HERE), seconds=time.time() - t0,
                  test_cfg=json.loads(json.dumps(mcfg.get('test_cfg'), default=str)),
                  test_pipeline=json.loads(json.dumps(dcfg.pipeline, default=str)))
    if args.limit is None:
        with open(os.path.join(RAW, f'det_{args.model}.json'), 'w') as f:
            json.dump(result, f, indent=1)
    print(json.dumps({k: v for k, v in result.items() if k not in ('test_pipeline',)}, default=str)[:2000])


if __name__ == '__main__':
    main()
