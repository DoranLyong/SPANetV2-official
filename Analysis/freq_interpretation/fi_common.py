"""Shared pieces of freq_interpretation: models / weights / images / masks of ../freq_robustness, Grid of
../srf_task_analysis. Nothing here changes model code."""
import importlib.util
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
AN = os.path.dirname(HERE)
FR = os.path.join(AN, 'freq_robustness')
sys.path.insert(0, FR)
import common as C  # noqa: E402,F401
import exp_cls as E  # noqa: E402,F401  (band_project, half_mask, l2n, fourier_basis, FOURIER_NORM, PGD_LOW_S)

_spec = importlib.util.spec_from_file_location('_srf', os.path.join(AN, 'srf_task_analysis', 'analyze_srf.py'))
SRF = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(SRF)          # Grid (conjugate-symmetric weights, rings, kernel DTFT)

DATA = os.path.join(HERE, 'data')
CACHE = os.path.join(HERE, 'cache')
NAME = {'spanetv2_s18_pure': 'SPANetV2-S18-pure', 'spanetv2_s18_hybrid': 'SPANetV2-S18-hybrid',
        'convformer_s18': 'ConvFormer-S18', 'caformer_s18': 'CAFormer-S18', 'dfformer_s18': 'DFFormer-S18',
        'cdfformer_s18': 'CDFFormer-S18', 'convnext_tiny': 'ConvNeXt-T', 'swin_tiny': 'Swin-T', 'deit_small': 'DeiT-S'}


def subset1k():
    """1,000 images: the first image of every class in the 5,000-image sample of freq_robustness
    (subset5k_seed0 is sorted by val index = class order, so its first 1,000 entries would cover classes 0-199 only)."""
    return C.subset5k()[::5]


def masks(dev):
    low = C.lowpass_mask(E.PGD_LOW_S, metric=E.MASK_METRIC)
    return {'full': None, 'high': E.half_mask(1 - low).to(dev), 'low': E.half_mask(low).to(dev)}


def save_json(name, obj):
    os.makedirs(DATA, exist_ok=True)
    with open(os.path.join(DATA, name), 'w') as f:
        json.dump(obj, f, indent=1)


def load_json(name):
    p = os.path.join(DATA, name)
    return json.load(open(p)) if os.path.exists(p) else None
