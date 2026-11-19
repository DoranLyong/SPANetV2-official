"""
Re-measure #params and MACs of SPANet / SPANetV2 and the comparison models
(ConvFormer, CAFormer, DFFormer, CDFFormer, optional GFNet-H) with one script,
one environment and one convention.

    python measure.py                 # all models -> results/results.json, results/env.json
    python measure.py --models SPANetV2-S18-pure ConvFormer-S18

Input 1x3x224x224, eval mode, CPU, randomly initialised weights. Values are MACs.

Per model:
  (ga) fvcore default                         FlopCountAnalysis(model, x).total()  (= notebook cell 6)
  (na) notebook handler (SPANet / SPANetV2)   cell 7: rfft_flop_jit on aten::fft_rfft2 / aten::fft_irfft2
  (a)  FFT       one transform = E * ceil(log2 N) MACs
                 E = #elements of the real-domain tensor (rfft*: input, irfft*: output;
                     complex fft*/ifft*: the complex output, same numel as the input),
                 N = product of the transformed axis sizes, read from the op's `dim` argument.
                 fftshift / ifftshift = 0.
  (b)  element-wise products in the frequency domain = aten::mul with a complex output, 1 per output element
  (c)  other element-wise arithmetic left uncounted by fvcore (add, sub, mul, div, pow, sqrt,
       square, gelu, sigmoid, softmax, mean, ...), 1 per output element
  Pure data-movement ops (clone, real, view_as_complex, ...) are 0.
  Ops on fvcore's ignore list that do arithmetic (aten::relu) stay 0 and are only logged.
Cross-check: torch.utils.flop_counter.FlopCounterMode (counts FLOPs = 2 x MACs).
"""
import argparse
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from numbers import Number
from typing import Any, List

import numpy as np
import torch
from fvcore.nn import FlopCountAnalysis
from fvcore.nn.jit_handles import get_shape
from torch.utils.flop_counter import FlopCounterMode

import zoo

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'results')
IMG = 224


# ---------------------------------------------------------------------------------------------
# (na) Handler copied verbatim from image_classification/FLOPs_count.ipynb, cell 2
#      (identical in the SPANet and SPANetV2 repositories).
def rfft_flop_jit(inputs: List[Any], outputs: List[Any]) -> Number:
    """
    Count flops for the rfft/rfftn operator.
    """
    input_shape = inputs[0].type().sizes()
    B, H, W, C = input_shape
    N = H * W
    flops = N * C * np.ceil(np.log2(N))
    return flops


# ---------------------------------------------------------------------------------------------
# One convention for all models
FFT_OPS = {  # op -> (real or complex transform, which tensor is the real-domain / reference tensor)
    'aten::fft_rfft2': ('real', 'input'),  'aten::fft_rfftn': ('real', 'input'),  'aten::fft_rfft': ('real', 'input'),
    'aten::fft_irfft2': ('real', 'output'), 'aten::fft_irfftn': ('real', 'output'), 'aten::fft_irfft': ('real', 'output'),
    'aten::fft_fft2': ('complex', 'output'), 'aten::fft_fftn': ('complex', 'output'), 'aten::fft_fft': ('complex', 'output'),
    'aten::fft_ifft2': ('complex', 'output'), 'aten::fft_ifftn': ('complex', 'output'), 'aten::fft_ifft': ('complex', 'output'),
}
FFT_SHIFT_OPS = {'aten::fft_fftshift', 'aten::fft_ifftshift'}
ELEMENTWISE_OPS = {
    'aten::add', 'aten::add_', 'aten::sub', 'aten::sub_', 'aten::rsub', 'aten::mul', 'aten::mul_',
    'aten::div', 'aten::div_', 'aten::pow', 'aten::sqrt', 'aten::rsqrt', 'aten::square', 'aten::exp',
    'aten::log', 'aten::neg', 'aten::abs', 'aten::reciprocal', 'aten::gelu', 'aten::sigmoid',
    'aten::tanh', 'aten::silu', 'aten::hardswish', 'aten::softmax', 'aten::mean', 'aten::sum',
}
REDUCTION_OPS = {'aten::mean', 'aten::sum'}
ZERO_OPS = {'aten::clone', 'aten::real', 'aten::imag', 'aten::view_as_complex', 'aten::view_as_real',
            'aten::expand_as', 'aten::repeat'}
LOG_ONLY_OPS = {'aten::relu', 'aten::relu_'}  # on fvcore's ignore list; kept at 0, numel logged


def ivalue(v):
    node = v.node()
    if node.kind() == 'prim::ListConstruct':
        return [ivalue(i) for i in node.inputs()]
    return v.toIValue()


def dtype_of(v):
    try:
        return v.type().scalarType()
    except Exception:
        return None


def scope_of(v):
    return v.node().scopeName().split('/')[-1]


class Convention:
    """fvcore op handles for (a), (b), (c); every call is logged."""

    def __init__(self):
        self.records = []

    def handles(self):
        h = {}
        for op in FFT_OPS:
            h[op] = self._fft(op)
        for op in FFT_SHIFT_OPS:
            h[op] = self._fixed(op, 'a', 'fftshift/ifftshift')
        for op in ELEMENTWISE_OPS:
            h[op] = self._elementwise(op)
        for op in ZERO_OPS:
            h[op] = self._fixed(op, 'zero', 'data movement')
        for op in LOG_ONLY_OPS:
            h[op] = self._fixed(op, 'ignored', 'fvcore ignore list')
        return h

    def _fft(self, op):
        kind, ref = FFT_OPS[op]

        def handle(inputs, outputs):
            x, out = inputs[0], outputs[0]
            t = x if ref == 'input' else out
            shape = get_shape(t)
            s_arg, dim_arg = ivalue(inputs[1]), ivalue(inputs[2])
            dims = list(range(len(shape))) if dim_arg is None else \
                ([dim_arg] if isinstance(dim_arg, int) else list(dim_arg))
            dims = [d % len(shape) for d in dims]
            E = math.prod(shape)
            N = math.prod(shape[d] for d in dims)
            L = math.ceil(math.log2(N))
            macs = E * L
            self.records.append(dict(
                op=op, cat='a', module=scope_of(out), count=macs, transform=kind,
                in_shape=get_shape(x), in_dtype=dtype_of(x), out_shape=get_shape(out), out_dtype=dtype_of(out),
                s=s_arg, dim=dim_arg, E_tensor=ref, E=E, N=N, ceil_log2_N=L))
            return macs
        return handle

    def _elementwise(self, op):
        def handle(inputs, outputs):
            out = outputs[0]
            shape = get_shape(out)
            n = math.prod(shape)
            dt = dtype_of(out)
            cat = 'b' if op in ('aten::mul', 'aten::mul_') and dt and dt.startswith('Complex') else 'c'
            rec = dict(op=op, cat=cat, module=scope_of(out), count=n, out_shape=shape, out_dtype=dt,
                       in_shapes=[get_shape(i) for i in inputs if dtype_of(i) is not None],
                       in_dtypes=[dtype_of(i) for i in inputs if dtype_of(i) is not None])
            if op in REDUCTION_OPS:
                rec['input_numel'] = math.prod(get_shape(inputs[0]))
            self.records.append(rec)
            return n
        return handle

    def _fixed(self, op, cat, note):
        def handle(inputs, outputs):
            out = outputs[0]
            shape = get_shape(out)
            self.records.append(dict(op=op, cat=cat, module=scope_of(out), count=0, note=note,
                                     out_shape=shape, out_dtype=dtype_of(out),
                                     numel=math.prod(shape) if shape is not None else None))
            return 0
        return handle


# ---------------------------------------------------------------------------------------------
def quiet(fca):
    fca.unsupported_ops_warnings(False)
    fca.uncalled_modules_warnings(False)
    fca.tracer_warnings('none')
    return fca


def pattern(name):
    return re.sub(r'(?<=\.)\d+(?=\.|$)|^\d+(?=\.|$)', '*', name) if name else '(root)'


def unsupported_by_module(fca, model):
    """Self (not inherited) unsupported-op calls per module, grouped by name pattern and class."""
    mods = dict(model.named_modules())
    out = Counter()
    for name, m in mods.items():
        own = Counter(fca.unsupported_ops(name))
        for child, _ in m.named_children():
            own.subtract(fca.unsupported_ops(f'{name}.{child}' if name else child))
        for op, n in own.items():
            if n > 0:
                out[(op, pattern(name), type(m).__name__)] += n
    rows = [dict(op=op, module=p, cls=c, calls=n) for (op, p, c), n in out.items()]
    return sorted(rows, key=lambda r: (r['op'], r['module']))


def aggregate(records):
    agg = defaultdict(lambda: dict(calls=0, elements=0, count=0))
    for r in records:
        key = (r['cat'], r['op'], r.get('out_dtype'), pattern(r['module']))
        a = agg[key]
        a['calls'] += 1
        if 'E' in r:  # FFT: E = #elements of the real-domain tensor (complex fft2/ifft2: the complex tensor)
            a['elements'] += r['E']
        else:
            a['elements'] += r.get('numel') or (math.prod(r['out_shape']) if r.get('out_shape') else 0)
        a['count'] += r['count']
        if 'input_numel' in r:
            a['input_numel'] = a.get('input_numel', 0) + r['input_numel']
    return [dict(cat=k[0], op=k[1], dtype=k[2], module=k[3], **v) for k, v in sorted(agg.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][3]))]


def fft_checks(records):
    """One record per FFT op kind (the first call) for the handler check in step 4."""
    seen, out = set(), []
    for r in records:
        if r['cat'] == 'a' and r['op'] in FFT_OPS and r['op'] not in seen:
            seen.add(r['op'])
            out.append({k: r[k] for k in ('op', 'module', 'transform', 'in_shape', 'in_dtype', 'out_shape',
                                           'out_dtype', 's', 'dim', 'E_tensor', 'E', 'N', 'ceil_log2_N', 'count')})
    return out


def measure(name, family, group, kind, paper, repo_key):
    model, ctor_stdout = zoo.build(name)
    torch.manual_seed(0)
    x = torch.rand(1, 3, IMG, IMG)
    res = dict(name=name, family=family, group=group, kind=kind, repo=repo_key,
               paper_params_M=paper[0], paper_macs_G=paper[1], ctor_stdout=ctor_stdout)

    res['params'] = sum(p.numel() for p in model.parameters())
    res['params_trainable'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    res['buffers'] = sum(b.numel() for b in model.buffers())

    # (ga) fvcore default -- notebook cell 6 (no torch.no_grad there)
    t0 = time.time()
    fca = quiet(FlopCountAnalysis(model, x))
    res['ga'] = fca.total()
    res['ga_by_operator'] = dict(fca.by_operator())
    res['unsupported_ops_raw'] = repr(fca.unsupported_ops())
    res['unsupported_ops'] = dict(fca.unsupported_ops())
    res['unsupported_by_module'] = unsupported_by_module(fca, model)

    # (na) notebook handler -- notebook cell 7 (inside torch.no_grad)
    if repo_key in ('spanet', 'spanetv2'):
        nb_log = []

        def logged(op):
            def handle(inputs, outputs):  # calls the notebook function unchanged, only logs its input
                val = rfft_flop_jit(inputs, outputs)
                B, H, W, C = inputs[0].type().sizes()
                nb_log.append(dict(op=op, module=scope_of(outputs[0]), in_shape=[B, H, W, C],
                                   in_dtype=dtype_of(inputs[0]), unpacked=dict(B=B, H=H, W=W, C=C),
                                   value=float(val)))
                return val
            return handle

        with torch.no_grad():
            fca_nb = quiet(FlopCountAnalysis(model, x))
            fca_nb.set_op_handle(**{op: logged(op) for op in ('aten::fft_rfft2', 'aten::fft_irfft2')})
            res['na'] = float(fca_nb.total())
        res['na_calls'] = len(nb_log)
        res['na_added'] = float(sum(r['value'] for r in nb_log))
        res['na_check'] = [next(r for r in nb_log if r['op'] == op)
                           for op in ('aten::fft_rfft2', 'aten::fft_irfft2') if any(r['op'] == op for r in nb_log)]
    else:
        res['na'] = None

    # (a)(b)(c) one convention
    conv = Convention()
    with torch.no_grad():
        fca_c = quiet(FlopCountAnalysis(model, x))
        fca_c.set_op_handle(**conv.handles())
        total_c = fca_c.total()
        left = dict(fca_c.unsupported_ops())
    if left:
        raise RuntimeError(f'{name}: ops without a handler in the new convention: {left}')
    for cat in ('a', 'b', 'c'):
        res[cat] = sum(r['count'] for r in conv.records if r['cat'] == cat)
    res['total'] = res['ga'] + res['a'] + res['b'] + res['c']
    assert math.isclose(total_c, res['total'], rel_tol=1e-12), (name, total_c, res['total'])
    res['convention_records'] = aggregate(conv.records)
    res['fft_checks'] = fft_checks(conv.records)

    # Cross-check: torch FlopCounterMode (FLOPs, 2 per multiply-add)
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        model(x)
    res['fcm_flops'] = fc.get_total_flops()
    res['fcm_by_op'] = {str(k): v for k, v in fc.get_flop_counts().get('Global', {}).items()}
    res['seconds'] = round(time.time() - t0, 1)
    return res


def git(repo, *args):
    return subprocess.run(['git', '-C', repo, *args], capture_output=True, text=True).stdout.strip()


def environment():
    import fvcore
    import timm
    env = dict(
        date=time.strftime('%Y-%m-%d %H:%M:%S %Z'),
        python=sys.version.split()[0], torch=torch.__version__, fvcore=getattr(fvcore, '__version__', '?'),
        timm=timm.__version__, numpy=np.__version__, device='cpu', torch_threads=torch.get_num_threads(),
        platform=platform.platform(), cpu=subprocess.run("lscpu | sed -n 's/^Model name: *//p'", shell=True,
                                                          capture_output=True, text=True).stdout.strip(),
        python_executable=sys.executable,
    )
    repos = {}
    for key, (root, rel) in zoo.FILES.items():
        repos[key] = dict(root=os.path.relpath(root, HERE), file=rel, commit=git(root, 'rev-parse', 'HEAD'),
                          remote=git(root, 'remote', 'get-url', 'origin'),
                          file_status=git(root, 'status', '--short', '--', rel) or 'clean',
                          repo_status=git(root, 'status', '--short') or 'clean')
    env['repos'] = repos
    return env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='*', default=None)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    env = environment()
    print(json.dumps(env, indent=2))
    with open(os.path.join(OUT, 'env.json'), 'w') as f:
        json.dump(env, f, indent=2)

    path = os.path.join(OUT, 'results.json')
    results = json.load(open(path)) if (args.models and os.path.exists(path)) else {}
    for name, family, group, kind, paper, repo_key, _ in zoo.MODELS:
        if args.models and name not in args.models:
            continue
        r = measure(name, family, group, kind, paper, repo_key)
        results[name] = r
        na = f"{r['na'] / 1e9:.3f}" if r['na'] is not None else '  -  '
        print(f"{name:22s} params {r['params'] / 1e6:7.3f}M | paper {paper[1]:5.1f} | (ga) {r['ga'] / 1e9:.3f} | "
              f"(na) {na} | a {r['a'] / 1e9:.3f} b {r['b'] / 1e9:.3f} c {r['c'] / 1e9:.3f} | total {r['total'] / 1e9:.3f} "
              f"| FCM/2 {r['fcm_flops'] / 2e9:.3f} | {r['seconds']}s", flush=True)
        with open(path, 'w') as f:
            json.dump(results, f, indent=1, default=str)


if __name__ == '__main__':
    main()
