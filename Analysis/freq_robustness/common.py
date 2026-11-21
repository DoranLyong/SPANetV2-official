"""Shared pieces for experiments 0, A, B, C (ImageNet-1K classification, 224x224).

Models are built from each official repository's own model file, loaded by path (importlib), unchanged:
  SPANetV2      SPANetV2-official  image_classification/models/spanetv2.py
  ConvFormer,   sail-sg/metaformer metaformer_baselines.py           (flops_benchmark/third_party, commit in report)
  CAFormer
  DFFormer,     okojoalg/dfformer  models/dfformer.py                (flops_benchmark/third_party)
  CDFFormer
  ConvNeXt      facebookresearch/ConvNeXt models/convnext.py         (third_party/, setup_third_party.sh)
  Swin          microsoft/Swin-Transformer models/swin_transformer.py
  DeiT          facebookresearch/deit models.py (timm VisionTransformer)
Images come from cache/ (make_cache.py): uint8 224x224 after each model's own Resize + CenterCrop; x/255 is the
[0,1] tensor of the model's pipeline. Filters / noise / attacks act on that [0,1] tensor; normalisation
((x - mean) / std) is the last step, inside `Classifier.forward`.
"""
import importlib.util
import json
import os
import sys
from functools import partial

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
TP = os.path.join(HERE, 'third_party')
FB_TP = os.path.join(REPO, 'flops_benchmark', 'third_party')
CACHE = os.path.join(HERE, 'cache')
DATA = os.path.join(HERE, 'data')
IMNET_MEAN, IMNET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

FILES = {
    'spanetv2': os.path.join(REPO, 'image_classification', 'models', 'spanetv2.py'),
    'metaformer': os.path.join(FB_TP, 'metaformer', 'metaformer_baselines.py'),
    'dfformer': os.path.join(FB_TP, 'dfformer', 'models', 'dfformer.py'),
    'convnext': os.path.join(TP, 'ConvNeXt', 'models', 'convnext.py'),
    'swin': os.path.join(TP, 'Swin-Transformer', 'models', 'swin_transformer.py'),
    'deit': os.path.join(TP, 'deit', 'models.py'),
}
_mods = {}


def load_module(key):
    if key not in _mods:
        name = f'_fr_{key}'
        spec = importlib.util.spec_from_file_location(name, FILES[key])
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        old, sys.dont_write_bytecode = sys.dont_write_bytecode, True     # do not write __pycache__ into the repos
        try:
            spec.loader.exec_module(mod)
        finally:
            sys.dont_write_bytecode = old
        _mods[key] = mod
    return _mods[key]


def _swin_tiny():
    # configs/swin/swin_tiny_patch4_window7_224.yaml (EMBED_DIM 96, DEPTHS 2-2-6-2, NUM_HEADS 3-6-12-24, WINDOW 7)
    return load_module('swin').SwinTransformer(img_size=224, patch_size=4, in_chans=3, num_classes=1000, embed_dim=96,
                                               depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size=7,
                                               mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0.,
                                               drop_path_rate=0.2, ape=False, patch_norm=True)


# key: display name, builder, weight key (download_weights.py), crop_pct, reference top-1 and its source
MODELS = {
    'spanetv2_s18_pure': dict(name='SPANetV2-S18-pure', group='SPANetV2',
                              build=lambda: load_module('spanetv2').spanetv2_s18_pure(), crop=1.0),
    'spanetv2_s18_hybrid': dict(name='SPANetV2-S18-hybrid', group='SPANetV2',
                                build=lambda: load_module('spanetv2').spanetv2_s18_hybrid(), crop=1.0),
    'convformer_s18': dict(name='ConvFormer-S18', group='MetaFormer',
                           build=lambda: load_module('metaformer').convformer_s18(), crop=1.0),
    'caformer_s18': dict(name='CAFormer-S18', group='MetaFormer',
                         build=lambda: load_module('metaformer').caformer_s18(), crop=1.0),
    'dfformer_s18': dict(name='DFFormer-S18', group='DFFormer',
                         build=lambda: load_module('dfformer').dfformer_s18(), crop=1.0),
    'cdfformer_s18': dict(name='CDFFormer-S18', group='DFFormer',
                          build=lambda: load_module('dfformer').cdfformer_s18(), crop=1.0),
    'convnext_tiny': dict(name='ConvNeXt-T', group='CNN', build=lambda: load_module('convnext').convnext_tiny(),
                          crop=0.875),     # convnext_tiny_1k_224.pth (file named in the task)
    'convnext_tiny_ema': dict(name='ConvNeXt-T (EMA file)', group='CNN',
                              build=lambda: load_module('convnext').convnext_tiny(), crop=0.875),
                              # convnext_tiny_1k_224_ema.pth: the file linked in the ConvNeXt README IN-1K table (82.1)
    'swin_tiny': dict(name='Swin-T', group='ViT', build=_swin_tiny, crop=0.875),
    'deit_small': dict(name='DeiT-S', group='ViT', build=lambda: load_module('deit').deit_small_patch16_224(),
                       crop=0.875),
}


def weight_path(key):
    with open(os.path.join(DATA, 'weights_manifest.json')) as f:
        return os.path.join(HERE, json.load(f)[key]['path'])


def load_weights(model, path):
    """torch.load(weights_only=True) first; returns (state_dict-key used, load info)."""
    info = {}
    try:
        ck = torch.load(path, map_location='cpu', weights_only=True)
        info['weights_only'] = True
    except Exception as e:
        info['weights_only'] = False
        info['weights_only_true_error'] = f'{type(e).__name__}: ' + ' '.join(str(e).split())[:300]
        ck = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(ck, dict) and 'model' in ck and isinstance(ck['model'], dict):
        sd, info['state_dict'] = ck['model'], "ckpt['model']"
    elif isinstance(ck, dict) and 'state_dict' in ck:
        sd, info['state_dict'] = ck['state_dict'], "ckpt['state_dict']"
    else:
        sd, info['state_dict'] = ck, 'flat'
    res = model.load_state_dict(sd, strict=False)
    info['missing'] = list(res.missing_keys)
    info['unexpected'] = list(res.unexpected_keys)
    return info


class Classifier(nn.Module):
    """[0,1] pixels (B,3,224,224) -> logits; the model's own mean/std normalisation is applied inside."""

    def __init__(self, key, weight=None):
        super().__init__()
        cfg = MODELS[key]
        self.key, self.cfg = key, cfg
        self.net = cfg['build']()
        self.load_info = load_weights(self.net, weight or weight_path(key))
        self.register_buffer('mean', torch.tensor(IMNET_MEAN).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(IMNET_STD).view(1, 3, 1, 1))
        self.eval()

    def forward(self, x01):
        return self.net((x01 - self.mean) / self.std)


def preprocessing(key):
    c = MODELS[key]['crop']
    return dict(resize=int(224 / c), resize_interpolation='bicubic (PIL)', center_crop=224, crop_pct=c,
                mean=IMNET_MEAN, std=IMNET_STD)


class ValCache:
    """uint8 cache of the 50k validation images for one crop_pct."""

    def __init__(self, crop):
        self.x = np.load(os.path.join(CACHE, f'val_crop{crop:.3f}.npy'), mmap_mode='r')
        self.y = np.load(os.path.join(CACHE, 'val_labels.npy'))

    def batches(self, idx, bs):
        for i in range(0, len(idx), bs):
            j = idx[i:i + bs]
            yield j, self.x[j], self.y[j]


def to01(u8, device):
    """(B,H,W,3) uint8 numpy -> (B,3,H,W) float32 in [0,1] on device."""
    return torch.from_numpy(np.array(u8)).to(device, non_blocking=True).permute(0, 3, 1, 2).float().div_(255)


def subset5k():
    return np.load(os.path.join(CACHE, 'subset5k_seed0.npy'))


def set_fp32():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = True


# ------------------------------------------------------------------------------------------------ frequency masks
def centered_dist(n=224, metric='chebyshev'):
    """Distance of each fftshifted DFT index from the centre (DC at index n//2)."""
    c = n // 2
    i = torch.arange(n).view(-1, 1) - c
    j = torch.arange(n).view(1, -1) - c
    if metric == 'chebyshev':
        return torch.maximum(i.abs(), j.abs()).float()
    if metric == 'min':
        return torch.minimum(i.abs(), j.abs()).float()
    if metric == 'euclid':
        return torch.sqrt((i ** 2 + j ** 2).float())
    raise ValueError(metric)


def lowpass_mask(S, n=224, metric='chebyshev'):
    """M_l^S: keep frequencies within S/2 of the centre (fftshifted layout)."""
    return (centered_dist(n, metric) <= S / 2).float()


def highpass_mask(S, n=224, metric='chebyshev'):
    """M_h^S: remove frequencies within (n - S)/2 of the centre."""
    return (centered_dist(n, metric) > (n - S) / 2).float()


def shifted_fft(x):
    """x (B,C,H,W) real float32 -> fftshifted complex64 spectrum per channel."""
    return torch.fft.fftshift(torch.fft.fft2(x.float()), dim=(-2, -1))


def apply_mask(x, mask, F=None):
    """x (B,C,H,W) real; mask (H,W) in fftshifted layout; per-channel FFT, multiply, inverse FFT, real part.
    F: precomputed shifted_fft(x) (reused when several masks are applied to the same batch)."""
    F = shifted_fft(x) if F is None else F
    y = torch.fft.ifft2(torch.fft.ifftshift(F * mask.to(F.device), dim=(-2, -1)))
    return y.real
