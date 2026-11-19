"""
Print the configuration actually built by the model code (read from the module objects, not from
the source text) and a per-component #params breakdown.

    python config_check.py > results/config_check.txt
"""
import re
from collections import OrderedDict

import torch.nn as nn

import zoo

TARGETS = ['SPANet-S', 'SPANet-M', 'SPANet-B',
           'SPANetV2-S18-pure', 'SPANetV2-S18-hybrid', 'SPANetV2-S36-pure', 'SPANetV2-S36-hybrid',
           'SPANetV2-M36-pure', 'SPANetV2-M36-hybrid', 'SPANetV2-B36-pure', 'SPANetV2-B36-hybrid',
           'ConvFormer-M36', 'ConvFormer-B36', 'CAFormer-M36', 'CAFormer-B36']


def n(m):
    return sum(p.numel() for p in m.parameters())


def component(name):
    """Group a parameter name into a coarse component."""
    if name.startswith(('downsample_layers', 'patch_embed')) or re.match(r'network\.[1357]\.', name):
        return 'downsample / stem'
    if name.startswith(('norm.', 'head.')):
        return 'final norm + head'
    if re.search(r'token_mixer\.filter_\d+$', name):
        return 'token_mixer: spectral filters (SRF params)'
    if re.search(r'token_mixer\.sps\.\d+\.filter$', name):
        return 'token_mixer: SPF filters (requires_grad=False)'
    if '.token_mixer.' in name:
        return 'token_mixer: other'
    if '.mlp.' in name:
        return 'mlp'
    if re.search(r'\.(norm1|norm2)\.', name):
        return 'block norms'
    if re.search(r'(res_scale|layer_scale)', name):
        return 'res/layer scale'
    return 'other'


def breakdown(model):
    out = OrderedDict()
    for name, p in model.named_parameters():
        c = component(name)
        out[c] = out.get(c, 0) + p.numel()
    return out


def describe_spanetv2(model):
    lines = []
    for i, stage in enumerate(model.stages):
        blk = stage[0]
        tm = blk.token_mixer
        dim = blk.norm1.weight.shape[0] if hasattr(blk.norm1, 'weight') else '?'
        d = f'  stage {i}: depth {len(stage)}, dim {dim}, token_mixer {type(tm).__name__}'
        if type(tm).__name__ == 'ExSPAM':
            ks = [getattr(tm, f'local_conv_{j + 1}').kernel_size for j in range(tm.num_heads)]
            fs = [tuple(getattr(tm, f'filter_{j + 1}').shape) for j in range(tm.num_heads)]
            d += (f', num_heads {tm.num_heads}, local_conv kernels {ks}, filters (C/heads, H, W//2+1) {fs}, '
                  f'proj {[type(m).__name__ for m in tm.proj]} expand {tm.proj[0].out_channels // tm.proj[0].in_channels}')
        else:
            d += f', num_heads {tm.num_heads}, head_dim {tm.head_dim}'
            if hasattr(tm, 'local_conv'):
                d += f', local_conv kernel {tm.local_conv.kernel_size}'
        mlp = blk.mlp
        d += f', mlp {mlp.fc1.in_features}->{mlp.fc1.out_features}->{mlp.fc2.out_features} ({type(mlp.act).__name__})'
        lines.append(d)
    h = model.head
    lines.append(f'  head: {type(h).__name__} '
                 + (f'{h.fc1.in_features}->{h.fc1.out_features}->{h.fc2.out_features}' if hasattr(h, 'fc1') else ''))
    return lines


def describe_spanet(model, stdout):
    lines = [f'  constructor stdout (print(self.lambs) per SPAM): {stdout.splitlines()[0] if stdout else "-"}']
    for idx in (0, 2, 4, 6):
        stage = model.network[idx]
        blk = stage[0]
        tm = blk.token_mixer
        f = [tuple(sp.filter.shape) for sp in tm.sps]
        uniq = sorted({round(v, 4) for v in tm.sps[0].filter.flatten().tolist()})
        lines.append(f'  stage {idx // 2}: depth {len(stage)}, dim {tm.proj_in.in_channels}, '
                     f'conv kernels {[c.kernel_size for c in tm.conv]}, #SPF {len(tm.sps)}, lambs {[round(float(l), 4) for l in tm.lambs]}, '
                     f'filter shapes {f}, distinct values in SPF[0] {uniq}')
    return lines


def main():
    for name in TARGETS:
        model, stdout = zoo.build(name)
        print(f'== {name}: {n(model) / 1e6:.3f} M params')
        if name.startswith('SPANetV2'):
            print('\n'.join(describe_spanetv2(model)))
        elif name.startswith('SPANet-'):
            print('\n'.join(describe_spanet(model, stdout)))
        for k, v in breakdown(model).items():
            print(f'    {k:45s} {v / 1e6:8.3f} M')
        print()


if __name__ == '__main__':
    main()
