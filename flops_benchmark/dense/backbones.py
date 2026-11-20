"""
Backbones for the dense-prediction re-measurement that are not registered in SPANetV2's
mmdet 3.x / mmseg 1.x copies. No model code is changed; this file only wraps and registers.

  convformer_{s18,s36}_feat, caformer_{s18,s36}_feat
      metaformer_baselines.py (sail-sg/metaformer) has no detection / segmentation backbone.
      `MetaFormerFeat` builds the classifier with its own factory function (random init) and returns the
      features of the four stages (out_indices 0-3), the same way SPANetV2's dense backbone does
      (object_detection/mmdet/models/backbones/spanetv2.py, SPANetV2.forward_features):
          x = downsample_layers[i](x); x = stages[i](x); out_i = norm{i}(x).permute(0, 3, 1, 2)
      with norm{i} = LayerNormWithoutBias(dims[i], eps=1e-6) (SPANetV2's default `norm_layers`,
      taken from metaformer_baselines.py). The classifier's final norm and head are not kept.

  SPANet (SPANet-S / SPANet-M)
      SPANet-official has no segmentation code. Its detection backbone
      (object_detection/mmdet/models/backbones/spanet.py, class `SPANet`, four-stage outputs with a
      GroupNorm per output) is loaded by path. Only its import lines are changed
      (patches/SPANet_det_backbone_mm2_imports.patch) so that it registers into mmseg 1.x.

  SegDFFormerS18/S36, SegCDFFormerS18/S36
      dfformer/segmentation/backbone/{dfformer,cdfformer}.py, imported unchanged
      (`from mmseg.models.builder import BACKBONES` still exists in mmseg 1.2.2).
"""
import contextlib
import importlib.util
import io
import os
import sys

import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
FB = os.path.dirname(HERE)                      # flops_benchmark/
if FB not in sys.path:
    sys.path.insert(0, FB)
import zoo  # noqa: E402

TP = zoo.TP
DFFORMER_ROOT = os.path.join(TP, 'dfformer')
SPANET_DET_BACKBONE = os.path.join(TP, 'SPANet-official', 'object_detection', 'mmdet', 'models', 'backbones', 'spanet.py')

FEAT_ARCHS = ['convformer_s18', 'convformer_s36', 'caformer_s18', 'caformer_s36']


class MetaFormerFeat(nn.Module):
    """metaformer_baselines.<arch>() -> 4-stage feature backbone (see module docstring)."""

    def __init__(self, arch, **unused):
        super().__init__()
        mf = zoo.load('metaformer')
        net = getattr(mf, arch)()          # random init, factory defaults
        self.arch = arch
        self.num_stage = net.num_stage
        self.downsample_layers = net.downsample_layers
        self.stages = net.stages
        dims = [d.conv.out_channels for d in net.downsample_layers]
        for i, d in enumerate(dims):
            setattr(self, f'norm{i}', mf.LayerNormWithoutBias(d, eps=1e-6))
        self.ignored_cfg_keys = sorted(unused)   # e.g. drop_path_rate / head_dropout / init_cfg from the config

    def forward(self, x):
        outs = []
        for i in range(self.num_stage):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            outs.append(getattr(self, f'norm{i}')(x).permute(0, 3, 1, 2).contiguous())
        return outs


def _feat_factory(arch):
    def build(**kwargs):
        return MetaFormerFeat(arch, **kwargs)
    build.__name__ = f'{arch}_feat'
    return build


def register_metaformer(registry):
    for arch in FEAT_ARCHS:
        name = f'{arch}_feat'
        if name not in registry.module_dict:
            registry.register_module(name=name, module=_feat_factory(arch))


def register_spanet_v1():
    """Load the (import-patched) SPANet detection backbone; it registers `SPANet` into mmseg's MODELS."""
    name = '_dense_spanet_v1_backbone'
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SPANET_DET_BACKBONE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    old, sys.dont_write_bytecode = sys.dont_write_bytecode, True   # SPANet-official tracks __pycache__/spanet.*.pyc
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = old
    return mod


def register_dfformer_seg():
    """Import dfformer/segmentation/backbone/{dfformer,cdfformer}.py unchanged (registers SegDFFormer*/SegCDFFormer*)."""
    if DFFORMER_ROOT not in sys.path:
        sys.path.insert(0, DFFORMER_ROOT)
    import segmentation.backbone.dfformer  # noqa: F401
    import segmentation.backbone.cdfformer  # noqa: F401


@contextlib.contextmanager
def captured_stdout():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield buf
