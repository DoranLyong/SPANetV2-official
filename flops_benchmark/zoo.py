"""
Model builders for the FLOPs / #params benchmark.

Each model file is loaded by its path (importlib), not as a package import:
  - SPANetV2 and DFFormer both keep their code in a package named `models`, so a
    normal import would collide.
  - SPANetV2's models/__init__.py would also be bypassed, so spanetv2_ver4.py is
    never imported.
Nothing in the model files is changed. The only edit in third_party/ is the GFNet
import path fix (patches/GFNet_numpy_import.patch).
"""
import contextlib
import importlib.util
import io
import os
import sys
from functools import partial

import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
TP = os.path.join(HERE, 'third_party')
SPANETV2_ROOT = os.path.dirname(HERE)  # SPANetV2-official/ (this folder lives inside it)

# key: (repo root, model file relative to the repo root)
FILES = {
    'spanet':     (os.path.join(TP, 'SPANet-official'), 'image_classification/spanet.py'),
    'spanetv2':   (SPANETV2_ROOT,                       'image_classification/models/spanetv2.py'),
    'metaformer': (os.path.join(TP, 'metaformer'),      'metaformer_baselines.py'),
    'dfformer':   (os.path.join(TP, 'dfformer'),        'models/dfformer.py'),
    'gfnet':      (os.path.join(TP, 'GFNet'),           'gfnet.py'),
}

_loaded = {}


def load(key):
    if key not in _loaded:
        repo, rel = FILES[key]
        name = f'_bench_{key}'
        spec = importlib.util.spec_from_file_location(name, os.path.join(repo, rel))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        _loaded[key] = mod
    return _loaded[key]


def _gfnet_h(arch):
    # Same constructor arguments as GFNet/infer.py and GFNet/main_gfnet.py (img_size 224).
    cfg = {
        'gfnet-h-s': dict(embed_dim=[96, 192, 384, 768], depth=[3, 3, 10, 3],
                          drop_path_rate=0.2, init_values=1e-5),
        'gfnet-h-b': dict(embed_dim=[96, 192, 384, 768], depth=[3, 3, 27, 3],
                          drop_path_rate=0.4, init_values=1e-6),
    }[arch]
    return load('gfnet').GFNetPyramid(
        img_size=224, patch_size=4, mlp_ratio=[4, 4, 4, 4],
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **cfg)


# name, family, size group, paper (params M, MACs G), builder
# Size groups follow the thesis table; SPANet-S/M/B sit next to the S18/S36/M36 pure models.
def _fn(key, fn):
    return lambda: getattr(load(key), fn)()


MODELS = [
    # ---- S18
    ('SPANet-S',           'SPANet',          'S18', 'pure',   (29, 4.6),  'spanet',     _fn('spanet', 'spanet_small')),
    ('SPANetV2-S18-pure',  'SPANetV2-pure',   'S18', 'pure',   (29, 4.2),  'spanetv2',   _fn('spanetv2', 'spanetv2_s18_pure')),
    ('ConvFormer-S18',     'ConvFormer',      'S18', 'pure',   (27, 3.9),  'metaformer', _fn('metaformer', 'convformer_s18')),
    ('DFFormer-S18',       'DFFormer',        'S18', 'pure',   (30, 3.8),  'dfformer',   _fn('dfformer', 'dfformer_s18')),
    ('SPANetV2-S18-hybrid', 'SPANetV2-hybrid', 'S18', 'hybrid', (27, 4.2), 'spanetv2',   _fn('spanetv2', 'spanetv2_s18_hybrid')),
    ('CAFormer-S18',       'CAFormer',        'S18', 'hybrid', (26, 4.1),  'metaformer', _fn('metaformer', 'caformer_s18')),
    ('CDFFormer-S18',      'CDFFormer',       'S18', 'hybrid', (30, 3.9),  'dfformer',   _fn('dfformer', 'cdfformer_s18')),
    # ---- S36
    ('SPANet-M',           'SPANet',          'S36', 'pure',   (42, 6.8),  'spanet',     _fn('spanet', 'spanet_medium')),
    ('SPANetV2-S36-pure',  'SPANetV2-pure',   'S36', 'pure',   (44, 8.1),  'spanetv2',   _fn('spanetv2', 'spanetv2_s36_pure')),
    ('ConvFormer-S36',     'ConvFormer',      'S36', 'pure',   (40, 7.6),  'metaformer', _fn('metaformer', 'convformer_s36')),
    ('DFFormer-S36',       'DFFormer',        'S36', 'pure',   (46, 7.4),  'dfformer',   _fn('dfformer', 'dfformer_s36')),
    ('SPANetV2-S36-hybrid', 'SPANetV2-hybrid', 'S36', 'hybrid', (41, 8.1), 'spanetv2',   _fn('spanetv2', 'spanetv2_s36_hybrid')),
    ('CAFormer-S36',       'CAFormer',        'S36', 'hybrid', (39, 8.0),  'metaformer', _fn('metaformer', 'caformer_s36')),
    ('CDFFormer-S36',      'CDFFormer',       'S36', 'hybrid', (45, 7.5),  'dfformer',   _fn('dfformer', 'cdfformer_s36')),
    # ---- M36
    ('SPANet-B',           'SPANet',          'M36', 'pure',   (76, 12.0), 'spanet',     _fn('spanet', 'spanet_base')),
    ('SPANetV2-M36-pure',  'SPANetV2-pure',   'M36', 'pure',   (61, 13.7), 'spanetv2',   _fn('spanetv2', 'spanetv2_m36_pure')),
    ('ConvFormer-M36',     'ConvFormer',      'M36', 'pure',   (57, 12.8), 'metaformer', _fn('metaformer', 'convformer_m36')),
    ('DFFormer-M36',       'DFFormer',        'M36', 'pure',   (65, 12.5), 'dfformer',   _fn('dfformer', 'dfformer_m36')),
    ('SPANetV2-M36-hybrid', 'SPANetV2-hybrid', 'M36', 'hybrid', (58, 13.6), 'spanetv2',  _fn('spanetv2', 'spanetv2_m36_hybrid')),
    ('CAFormer-M36',       'CAFormer',        'M36', 'hybrid', (56, 13.2), 'metaformer', _fn('metaformer', 'caformer_m36')),
    ('CDFFormer-M36',      'CDFFormer',       'M36', 'hybrid', (64, 12.7), 'dfformer',   _fn('dfformer', 'cdfformer_m36')),
    # ---- B36
    ('SPANetV2-B36-pure',  'SPANetV2-pure',   'B36', 'pure',   (100, 24.3), 'spanetv2',  _fn('spanetv2', 'spanetv2_b36_pure')),
    ('ConvFormer-B36',     'ConvFormer',      'B36', 'pure',   (100, 22.6), 'metaformer', _fn('metaformer', 'convformer_b36')),
    ('DFFormer-B36',       'DFFormer',        'B36', 'pure',   (115, 22.1), 'dfformer',  _fn('dfformer', 'dfformer_b36')),
    ('SPANetV2-B36-hybrid', 'SPANetV2-hybrid', 'B36', 'hybrid', (100, 23.9), 'spanetv2', _fn('spanetv2', 'spanetv2_b36_hybrid')),
    ('CAFormer-B36',       'CAFormer',        'B36', 'hybrid', (99, 23.2), 'metaformer', _fn('metaformer', 'caformer_b36')),
    ('CDFFormer-B36',      'CDFFormer',       'B36', 'hybrid', (113, 22.5), 'dfformer',  _fn('dfformer', 'cdfformer_b36')),
    # ---- (optional) GFNet-H, official repo. GFNet-H-S++ / H-L++ are not in that repo.
    ('GFNet-H-S',          'GFNet-H',         'GFNet', '-',    (32, 4.6),  'gfnet',      lambda: _gfnet_h('gfnet-h-s')),
    ('GFNet-H-B',          'GFNet-H',         'GFNet', '-',    (54, 8.6),  'gfnet',      lambda: _gfnet_h('gfnet-h-b')),
]


def build(name):
    """Build a randomly initialised model in eval mode on CPU. Constructor prints are captured."""
    entry = next(m for m in MODELS if m[0] == name)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        model = entry[-1]()
    model.eval()
    return model, buf.getvalue()
