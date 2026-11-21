#!/usr/bin/env python
"""D1 / D2: single-scale ADE20K val inference (mmseg 1.x copy of SPANetV2-official), predictions saved for D metrics.

    python seg_eval.py --model upernet_spanetv2_s18_pure [--weight PATH]      # GPU via CUDA_VISIBLE_DEVICES
    python seg_eval.py --list

Writes cache/preds/seg/<model>/<image>.png (uint8 class index 0..149, i.e. reduce_zero_label space) and
data/raw/seg_<model>.json (mIoU from this run, load info, config summary).

Models
  SPANetV2       config = the training config saved in the checkpoint (meta['cfg']), data_root replaced.
  ConvNeXt-T/S   facebookresearch/ConvNeXt semantic_segmentation/backbone/convnext.py (official, unchanged) +
                 mmseg 1.x port of configs/convnext/upernet_convnext_{tiny,small}_512_160k_ade20k_ss.py.
  Swin-T/S       SwinTransformer/Swin-Transformer-Semantic-Segmentation mmseg/models/backbones/swin_transformer.py
                 (official, unchanged) + port of configs/swin/upernet_swin_{tiny,small}_patch4_window7_512x512_160k_ade20k.py.
  PoolFormer     sail-sg/poolformer models/poolformer.py class PoolFormer with the arguments of poolformer_s{24,36}_feat
                 (official, unchanged) + mmseg model-zoo config fpn_poolformer_s{24,36}_8xb4-40k_ade20k-512x512.py
                 and the mmseg model-zoo weights (the repo's own weights are on Google Drive only).
The official backbone files were written for mmcv 1.x / mmseg 0.x. They are imported unchanged; while they are
imported, sys.modules temporarily holds small stand-ins for `mmcv_custom`, `mmseg.models.builder` (a registry that
registers the class as 'Official<ClassName>' in mmseg 1.x MODELS) and `mmseg.utils.get_root_logger`
(-> mmengine MMLogger). Nothing they define is used at inference except the backbone classes.
"""
import argparse
import copy
import importlib.util
import json
import os
import sys
import time
import types

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
TP = os.path.join(HERE, 'third_party')
ADE = os.path.join(HERE, 'cache', 'datasets', 'ade20k', 'ADEChallengeData2016')
PREDS = os.path.join(HERE, 'cache', 'preds', 'seg')
RAW = os.path.join(HERE, 'data', 'raw')

MEAN, STD = [123.675, 116.28, 103.53], [58.395, 57.12, 57.375]
NORM = dict(type='SyncBN', requires_grad=True)
TEST_RESIZE = [dict(type='LoadImageFromFile'), dict(type='Resize', scale=(2048, 512), keep_ratio=True),
               dict(type='LoadAnnotations', reduce_zero_label=True), dict(type='PackSegInputs')]
TEST_RESIZE_MULT = [dict(type='LoadImageFromFile'), dict(type='Resize', scale=(2048, 512), keep_ratio=True),
                    dict(type='ResizeToMultiple', size_divisor=32),
                    dict(type='LoadAnnotations', reduce_zero_label=True), dict(type='PackSegInputs')]


def setup_mmseg():
    sys.path.insert(0, os.path.join(REPO, 'semantic_segmentation'))
    import mmseg.datasets  # noqa: F401
    import mmseg.models  # noqa: F401
    from mmengine.registry import init_default_scope
    init_default_scope('mmseg')


class _OfficialRegistry:
    """Stand-in for mmseg 0.x BACKBONES: registers the decorated class as 'Official<Name>' in mmseg 1.x MODELS."""

    def register_module(self, name=None, force=False, module=None):
        from mmseg.registry import MODELS

        def deco(cls):
            MODELS.register_module(name=f'Official{name or cls.__name__}', module=cls, force=True)
            return cls
        return deco(module) if module is not None else deco


def load_official(path, modname, package=None):
    from mmengine.logging import MMLogger
    from mmengine.runner import load_checkpoint
    import mmseg.utils
    stubs = {
        'mmcv_custom': types.SimpleNamespace(load_checkpoint=load_checkpoint),
        'mmseg.models.builder': types.SimpleNamespace(BACKBONES=_OfficialRegistry()),
    }
    saved = {k: sys.modules.get(k) for k in stubs}
    had_logger = hasattr(mmseg.utils, 'get_root_logger')
    sys.modules.update(stubs)
    if not had_logger:
        mmseg.utils.get_root_logger = lambda *a, **k: MMLogger.get_current_instance()
    try:
        spec = importlib.util.spec_from_file_location(modname, path)
        mod = importlib.util.module_from_spec(spec)
        if package:
            mod.__package__ = package
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
        if not had_logger:
            del mmseg.utils.get_root_logger
    return mod


def register_poolformer():
    """PoolFormer (official class) with the arguments of poolformer_s{24,36}_feat; the *_feat classes themselves are
    only defined when mmcv 1.x mmseg and mmdet both import, so the same arguments are passed here."""
    from mmseg.registry import MODELS
    pf = load_official(os.path.join(TP, 'poolformer', 'models', 'poolformer.py'), '_fr_poolformer')
    presets = {'s24': dict(layers=[4, 4, 12, 4]), 's36': dict(layers=[6, 6, 18, 6], layer_scale_init_value=1e-6)}
    for arch, p in presets.items():
        def build(arch=arch, p=p, **kw):
            p = dict(p)
            layers = p.pop('layers')
            return pf.PoolFormer(layers, embed_dims=[64, 128, 320, 512], mlp_ratios=[4, 4, 4, 4],
                                 downsamples=[True, True, True, True], fork_feat=True, **p, **kw)
        MODELS.register_module(name=f'OfficialPoolFormer_{arch}_feat', module=build, force=True)


def upernet(backbone, in_channels, mode):
    return dict(
        type='EncoderDecoder',
        data_preprocessor=dict(type='SegDataPreProcessor', mean=MEAN, std=STD, bgr_to_rgb=True, pad_val=0,
                               seg_pad_val=255),
        backbone=backbone,
        decode_head=dict(type='UPerHead', in_channels=in_channels, in_index=[0, 1, 2, 3], pool_scales=(1, 2, 3, 6),
                         channels=512, dropout_ratio=0.1, num_classes=150, norm_cfg=NORM, align_corners=False,
                         loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
        auxiliary_head=dict(type='FCNHead', in_channels=in_channels[2], in_index=2, channels=256, num_convs=1,
                            concat_input=False, dropout_ratio=0.1, num_classes=150, norm_cfg=NORM,
                            align_corners=False,
                            loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        train_cfg=dict(),
        test_cfg=mode)


def fpn_poolformer(arch):
    return dict(
        type='EncoderDecoder',
        data_preprocessor=dict(type='SegDataPreProcessor', mean=MEAN, std=STD, bgr_to_rgb=True, pad_val=0,
                               seg_pad_val=255),
        backbone=dict(type=f'OfficialPoolFormer_{arch}_feat'),
        neck=dict(type='FPN', in_channels=[64, 128, 320, 512], out_channels=256, num_outs=4),
        decode_head=dict(type='FPNHead', in_channels=[256, 256, 256, 256], in_index=[0, 1, 2, 3],
                         feature_strides=[4, 8, 16, 32], channels=128, dropout_ratio=0.1, num_classes=150,
                         norm_cfg=NORM, align_corners=False,
                         loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
        train_cfg=dict(),
        test_cfg=dict(mode='whole'))


SLIDE = dict(mode='slide', crop_size=(512, 512), stride=(341, 341))
CNX = dict(type='OfficialConvNeXt', in_chans=3, layer_scale_init_value=1.0, out_indices=[0, 1, 2, 3])
SWIN = dict(type='OfficialSwinTransformer', embed_dim=96, num_heads=[3, 6, 12, 24], window_size=7, mlp_ratio=4.,
            qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.3, ape=False,
            patch_norm=True, out_indices=(0, 1, 2, 3), use_checkpoint=False)

# key: (display name, setting, pair group, model cfg or None (= checkpoint cfg), test pipeline or None, reference)
SEG_MODELS = {
    'upernet_spanetv2_s18_pure': ('SPANetV2-S18-pure', 'upernet', 'S18', None, None, None),
    'upernet_spanetv2_s18_hybrid': ('SPANetV2-S18-hybrid', 'upernet', 'S18', None, None, None),
    'upernet_spanetv2_s36_pure': ('SPANetV2-S36-pure', 'upernet', 'S36', None, None, None),
    'upernet_spanetv2_s36_hybrid': ('SPANetV2-S36-hybrid', 'upernet', 'S36', None, None, None),
    'upernet_convnext_tiny': ('ConvNeXt-T', 'upernet', 'S18',
                              upernet(dict(CNX, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0.4),
                                      [96, 192, 384, 768], SLIDE), TEST_RESIZE,
                              dict(mIoU_ss=46.0, mIoU_ms=46.7, src='facebookresearch/ConvNeXt semantic_segmentation README')),
    'upernet_convnext_small': ('ConvNeXt-S', 'upernet', 'S36',
                               upernet(dict(CNX, depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], drop_path_rate=0.3),
                                       [96, 192, 384, 768], SLIDE), TEST_RESIZE,
                               dict(mIoU_ss=48.7, mIoU_ms=49.6, src='facebookresearch/ConvNeXt semantic_segmentation README')),
    'upernet_swin_tiny': ('Swin-T', 'upernet', 'S18',
                          upernet(dict(SWIN, depths=[2, 2, 6, 2]), [96, 192, 384, 768], dict(mode='whole')),
                          TEST_RESIZE,
                          dict(mIoU_ss=44.51, mIoU_ms=45.81, src='Swin-Transformer-Semantic-Segmentation README')),
    'upernet_swin_small': ('Swin-S', 'upernet', 'S36',
                           upernet(dict(SWIN, depths=[2, 2, 18, 2]), [96, 192, 384, 768], dict(mode='whole')),
                           TEST_RESIZE,
                           dict(mIoU_ss=47.64, mIoU_ms=49.47, src='Swin-Transformer-Semantic-Segmentation README')),
    'fpn_spanetv2_s18_pure': ('SPANetV2-S18-pure', 'fpn', 'S18', None, None, None),
    'fpn_spanetv2_s18_hybrid': ('SPANetV2-S18-hybrid', 'fpn', 'S18', None, None, None),
    'fpn_spanetv2_s36_pure': ('SPANetV2-S36-pure', 'fpn', 'S36', None, None, None),
    'fpn_spanetv2_s36_hybrid': ('SPANetV2-S36-hybrid', 'fpn', 'S36', None, None, None),
    'fpn_poolformer_s24': ('PoolFormer-S24', 'fpn', 'S18', fpn_poolformer('s24'), TEST_RESIZE_MULT,
                           dict(mIoU_ss=40.36, mIoU_ss_without_resize_to_multiple=40.12,
                                src='mmseg model zoo configs/poolformer README (mIoU*: Resize + ResizeToMultiple)',
                                repo_readme=40.3)),
    'fpn_poolformer_s36': ('PoolFormer-S36', 'fpn', 'S36', fpn_poolformer('s36'), TEST_RESIZE_MULT,
                           dict(mIoU_ss=41.81, mIoU_ss_without_resize_to_multiple=41.61,
                                src='mmseg model zoo configs/poolformer README (mIoU*: Resize + ResizeToMultiple)',
                                repo_readme=42.0)),
}


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
    """SPANetV2 checkpoints carry their mmengine training config in meta['cfg']."""
    import tempfile
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
    ap.add_argument('--model', choices=list(SEG_MODELS))
    ap.add_argument('--weight', default=None)
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--limit', type=int, default=None, help='first N images only (smoke test)')
    args = ap.parse_args()
    if args.list:
        print('\n'.join(SEG_MODELS))
        return
    setup_mmseg()
    from mmengine.dataset import pseudo_collate
    from mmengine.model import revert_sync_batchnorm
    from mmseg.registry import DATASETS, MODELS
    load_official(os.path.join(TP, 'ConvNeXt', 'semantic_segmentation', 'backbone', 'convnext.py'), '_fr_convnext_seg')
    load_official(os.path.join(TP, 'Swin-Transformer-Semantic-Segmentation', 'mmseg', 'models', 'backbones',
                               'swin_transformer.py'), 'mmseg.models.backbones._fr_swin_seg',
                  package='mmseg.models.backbones')
    register_poolformer()

    name, setting, pair, mcfg, pipeline, ref = SEG_MODELS[args.model]
    wpath = args.weight or weight_path(args.model)
    ck, info = load_weights(wpath)
    cfg_src = 'given in seg_eval.py (port of the official config)'
    if mcfg is None:
        cfg = ckpt_config(ck)
        mcfg = copy.deepcopy(cfg.model)
        mcfg['backbone'].pop('init_cfg', None)
        pipeline = copy.deepcopy(cfg.test_dataloader.dataset.pipeline)
        cfg_src = "checkpoint meta['cfg']"
    from mmengine.config import ConfigDict
    mcfg = ConfigDict(copy.deepcopy(mcfg))              # attribute access (test_cfg.mode) as with a Config file
    sd = ck['state_dict'] if 'state_dict' in ck else ck
    torch.manual_seed(0)
    model = MODELS.build(mcfg)
    res = model.load_state_dict(sd, strict=False)
    info.update(missing=list(res.missing_keys), unexpected=list(res.unexpected_keys),
                state_dict="ckpt['state_dict']" if 'state_dict' in ck else 'flat')
    meta = ck.get('meta', {}) if isinstance(ck, dict) else {}
    info['meta'] = {k: (v if isinstance(v, (int, float, str)) else str(v))
                    for k, v in meta.items() if k in ('epoch', 'iter', 'mmseg_version', 'mmcv_version', 'time')}
    del ck, sd
    model = revert_sync_batchnorm(model).cuda().eval()

    ds = DATASETS.build(dict(type='ADE20KDataset', data_root=ADE, reduce_zero_label=True,
                             data_prefix=dict(img_path='images/validation', seg_map_path='annotations/validation'),
                             pipeline=pipeline, test_mode=True))
    out_dir = os.path.join(PREDS, args.model)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(RAW, exist_ok=True)
    import cv2
    n = len(ds) if args.limit is None else args.limit
    inter = np.zeros(150, np.int64)
    union = np.zeros(150, np.int64)
    area_gt = np.zeros(150, np.int64)
    t0 = time.time()
    with torch.no_grad():
        for i in range(n):
            out = model.test_step(pseudo_collate([ds[i]]))[0]
            pred = out.pred_sem_seg.data[0].cpu().numpy().astype(np.uint8)
            gt = out.gt_sem_seg.data[0].cpu().numpy()
            assert pred.shape == gt.shape, (pred.shape, gt.shape)
            stem = os.path.splitext(os.path.basename(out.img_path))[0]
            cv2.imwrite(os.path.join(out_dir, stem + '.png'), pred)
            v = gt != 255
            p, g = pred[v].astype(np.int64), gt[v].astype(np.int64)
            ai = np.bincount(p[p == g], minlength=150)[:150]
            ap_ = np.bincount(p, minlength=150)[:150]
            ag = np.bincount(g, minlength=150)[:150]
            inter += ai
            union += ap_ + ag - ai
            area_gt += ag
            if i % 200 == 0:
                print(f'{i}/{n} {time.time() - t0:.0f}s', flush=True)
    iou = inter / np.maximum(union, 1)
    valid = union > 0                                  # mmseg IoUMetric: IoU = nan when union = 0, then nanmean
    result = dict(model=args.model, name=name, setting=setting, pair=pair, n_images=n,
                  mIoU=float(iou[valid].mean() * 100), aAcc=float(inter.sum() / max(area_gt.sum(), 1) * 100),
                  per_class_IoU=[float(x) for x in iou], reference=ref, config_source=cfg_src,
                  test_cfg=json.loads(json.dumps(mcfg.get('test_cfg'), default=str)),
                  test_pipeline=json.loads(json.dumps(pipeline, default=str)), load_info=info,
                  weight=os.path.relpath(wpath, HERE), seconds=time.time() - t0)
    if args.limit is None:
        with open(os.path.join(RAW, f'seg_{args.model}.json'), 'w') as f:
            json.dump(result, f, indent=1)
    print(json.dumps({k: v for k, v in result.items() if k not in ('per_class_IoU', 'test_pipeline')}, default=str)[:1500])


if __name__ == '__main__':
    main()
