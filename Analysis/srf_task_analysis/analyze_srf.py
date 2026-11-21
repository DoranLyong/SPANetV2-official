#!/usr/bin/env python
"""
SRF mask analysis of SPANetV2 checkpoints (ImageNet-1K vs. COCO / ADE20K fine-tuned backbones). Read-only.

The SRF of ExSPAM is the parameter M = token_mixer.filter_{h}, shape (C/4, H, W//2+1), applied as
psi = sigmoid(M) times rfft2(local_conv_{h}(ctx_h), norm='ortho') (object_detection/mmdet/models/backbones/
spanetv2.py, ExSPAM.forward). Everything here is computed from the stored parameters; no data, no forward pass.

Usage (one model per call; checkpoint paths are arguments):

    python analyze_srf.py analyze --model S18pure \
        --ckpt IN1K=<in1k.pth> --ckpt COCO/MaskRCNN=<...> --ckpt COCO/Cascade=<...> \
        --ckpt ADE20K/FPN=<...> --ckpt ADE20K/UperNet=<...>
    python analyze_srf.py merge            # data/per_model/* -> data/srf_summary.csv, srf_radial.npz, fit_decay.csv

`run_all.sh` lists the exact calls used for report.md. The first --ckpt must be the IN1K reference.

Conventions (all on the stored 224-input grid; stage s has H = W = 56 / 2**(s-1)):
  ky = fftfreq(H), kx = rfftfreq(W) [cycles/pixel];  rho = sqrt(ky^2 + kx^2) / 0.5.
  Every average over frequency is the full-grid average of the rfft half grid: column kx = 0 (and the Nyquist
  column when W is even) has weight 1, every other column weight 2 (conjugate symmetry).
  Bands: DC rho = 0 | low 0 < rho <= 0.25 | mid 0.25 < rho <= 0.5 | high rho > 0.5.
  Radial rings: integer frequency radius r = round(sqrt(iy^2 + ix^2)) with iy = ky*H, ix = kx*W; ring centre
  rho = 2r/H.
  Statistics "across channels" pool all blocks of the stage: population = n_blocks x C/4 (block, channel) pairs.
  Kernel response |K^| is the DTFT of local_conv_h.weight on the same frequency grid (= rfft2 of the kernel
  zero-padded to H x W when k <= H; for k > H, i.e. stage 4 / kernel 9 on the 7 x 7 grid, it equals the
  rfft2 of the kernel wrapped modulo H). Bias is not included.
"""
import argparse
import glob
import hashlib
import json
import os
import re
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')
FIGS = os.path.join(HERE, 'figs')
REPO = os.path.dirname(os.path.dirname(HERE))

FILTER_RE = re.compile(r'^(?P<prefix>.*?)stages\.(?P<s>\d+)\.(?P<b>\d+)\.token_mixer\.filter_(?P<h>[1-4])$')
KSIZE = {1: 3, 2: 5, 3: 7, 4: 9}
BANDS = ('DC', 'low', 'mid', 'high')
EPS = 1e-12


# ----------------------------------------------------------------------------------------------- loading / step 0
def sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_ckpt(path):
    """torch.load with weights_only=True first; fall back to weights_only=False (author's own release files)."""
    info = {}
    try:
        ck = torch.load(path, map_location='cpu', weights_only=True)
        info['weights_only'] = True
    except Exception as e:  # mmengine metadata (numpy / mmengine objects) is not allow-listed
        msg = str(e).strip().splitlines()
        info['weights_only'] = False
        info['weights_only_true_error'] = f'{type(e).__name__}: ' + ' '.join(m.strip() for m in msg[:3])[:400]
        ck = torch.load(path, map_location='cpu', weights_only=False)
    return ck, info


def select_state_dict(ck):
    """Same rule as SPANetV2.init_weights (object_detection/.../spanetv2.py):
    'state_dict' -> ckpt['state_dict'], elif 'model' -> ckpt['model'], else the checkpoint itself."""
    if 'state_dict' in ck:
        return ck['state_dict'], "ckpt['state_dict']"
    if 'model' in ck:
        return ck['model'], "ckpt['model']"
    return ck, 'ckpt (flat state_dict)'


def collect(sd):
    """-> filters {(stage, block, head): M}, kernels {(stage, block, head): K}, prefix, filter key names."""
    filters, kernels, prefixes, names = {}, {}, set(), []
    for k, v in sd.items():
        m = FILTER_RE.match(k)
        if not m:
            continue
        s, b, h = int(m['s']) + 1, int(m['b']), int(m['h'])       # stage 1-based
        prefixes.add(m['prefix'])
        names.append(k)
        filters[(s, b, h)] = v.detach().double().numpy()
        kk = k[: -len(f'filter_{h}')] + f'local_conv_{h}.weight'
        kernels[(s, b, h)] = sd[kk].detach().double().numpy()[:, 0]  # (C/4, k, k)
    assert len(prefixes) == 1, prefixes
    return filters, kernels, prefixes.pop(), names


def mm_meta(ck):
    """Training metadata saved by mmengine (epoch / iter / config text), if present."""
    meta = ck.get('meta') if isinstance(ck, dict) else None
    if not isinstance(meta, dict):
        return None
    out = {k: meta[k] for k in ('epoch', 'iter', 'mmengine_version', 'mmdet_version', 'mmseg_version',
                                'time', 'seed', 'experiment_name') if k in meta}
    for k in ('epoch', 'iter', 'seed'):
        if k in out:
            out[k] = int(out[k])
    if 'cfg' in meta:
        out['cfg_text'] = meta['cfg']
    return out


def cfg_facts(cfg):
    """The training settings that matter here, from an mmengine Config."""
    bb = cfg.model.backbone
    ow = cfg.optim_wrapper
    out = dict(
        backbone_type=bb.get('type'), init_cfg=bb.get('init_cfg'), drop_path_rate=bb.get('drop_path_rate'),
        optim_wrapper_type=ow.get('type'), constructor=ow.get('constructor', 'DefaultOptimWrapperConstructor'),
        paramwise_cfg=ow.get('paramwise_cfg'), optimizer=ow.get('optimizer'),
        accumulative_counts=ow.get('accumulative_counts', 1), clip_grad=ow.get('clip_grad'),
        param_scheduler=cfg.get('param_scheduler'), train_cfg=cfg.get('train_cfg'),
        train_batch_size=cfg.train_dataloader.get('batch_size'), auto_scale_lr=cfg.get('auto_scale_lr'),
        load_from=cfg.get('load_from'), resume=cfg.get('resume'))
    return json.loads(json.dumps(out, default=str))


def load_cfg_file(path):
    from mmengine.config import Config
    # import_custom_modules=False: some configs have custom_imports=['mmdet.models.backbones.spanetv2'], not needed here
    return Config.fromfile(path, import_custom_modules=False)


def parse_cfg(cfg_text):
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 'cfg.py')
        with open(p, 'w') as f:
            f.write(cfg_text)
        return cfg_facts(load_cfg_file(p))


def optimizer_facts(ck, sd_keys, filter_names):
    """Param-group facts (lr_scale / lr / weight_decay / step) of the filter parameters from the saved optimizer."""
    opt = ck.get('optimizer') if isinstance(ck, dict) else None
    if not isinstance(opt, dict) or 'param_groups' not in opt:
        return None
    groups, state = opt['param_groups'], opt.get('state', {})
    idx2name = {}
    if all('param_names' in g for g in groups):
        for g in groups:
            for i, n in zip(g['params'], g['param_names']):
                idx2name[i] = n
        how = "param_groups[*]['param_names']"
    else:  # DefaultOptimWrapperConstructor: one group in model.parameters() order = state_dict order w/o buffers
        buf = ('running_mean', 'running_var', 'num_batches_tracked')
        pnames = [k for k in sd_keys if not k.endswith(buf)]
        ids = [i for g in groups for i in g['params']]
        if len(ids) != len(pnames):
            return dict(error=f'cannot map {len(ids)} optimizer ids to {len(pnames)} state_dict parameters')
        idx2name = dict(zip(ids, pnames))
        how = 'model.parameters() order (state_dict keys without BN buffers)'
    name2group = {}
    for g in groups:
        for i in g['params']:
            name2group[idx2name[i]] = g
    rows, steps = {}, set()
    fset = set(filter_names)
    for i, n in idx2name.items():
        if n not in fset:
            continue
        g = name2group[n]
        st = state.get(i, {})
        step = st.get('step')
        step = int(step.item() if torch.is_tensor(step) else step) if step is not None else None
        steps.add(step)
        rows[n] = dict(group=g.get('group_name'), lr_scale=g.get('lr_scale', 1.0), lr_at_save=g.get('lr'),
                       initial_lr=g.get('initial_lr'), weight_decay=g.get('weight_decay'), step=step,
                       exp_avg_abs_mean=float(st['exp_avg'].abs().mean()) if 'exp_avg' in st else None,
                       exp_avg_sq_mean=float(st['exp_avg_sq'].mean()) if 'exp_avg_sq' in st else None)
    missing = sorted(fset - set(rows))
    return dict(mapping=how, n_groups=len(groups), filters=rows, filters_not_in_optimizer=missing,
                adam_steps=sorted(s for s in steps if s is not None),
                optimizer_defaults={k: v for k, v in groups[0].items() if k in ('betas', 'eps', 'amsgrad')})


def lr_schedule(cfg, meta):
    """Base learning rate (lr_scale = 1) at every optimizer step of the saved run, from the run's own
    param_scheduler simulated with mmengine's scheduler classes (iteration-based ones step after every
    iteration, epoch-based ones after every epoch; one optimizer step per `accumulative_counts` iterations)."""
    from mmengine.optim.scheduler import LinearLR, MultiStepLR, PolyLR
    total_iters = meta['iter']
    accum = int(cfg.get('accumulative_counts') or 1)
    tc = cfg['train_cfg'] or {}
    epoch_based = 'max_epochs' in tc or tc.get('type') == 'EpochBasedTrainLoop'
    ipe = total_iters // meta['epoch'] if epoch_based and meta.get('epoch') else None
    base_lr = cfg['optimizer']['lr']
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=base_lr)
    it_sch, ep_sch = [], []
    cls = dict(LinearLR=LinearLR, MultiStepLR=MultiStepLR, PolyLR=PolyLR)
    for s in cfg['param_scheduler']:
        s = dict(s)
        sch = cls[s.pop('type')](opt, **s)
        (ep_sch if s.get('by_epoch', True) else it_sch).append(sch)
    lrs = []
    for it in range(total_iters):
        if (it + 1) % accum == 0:
            lrs.append(opt.param_groups[0]['lr'])
        for sch in it_sch:
            sch.step()
        if ipe and (it + 1) % ipe == 0:
            for sch in ep_sch:
                sch.step()
    info = dict(total_iters=total_iters, iters_per_epoch=ipe, accumulative_counts=accum,
                n_opt_steps_simulated=len(lrs), base_lr=base_lr, sum_base_lr=float(np.sum(lrs)),
                lr_first=lrs[0], lr_last=lrs[-1], lr_after_last_scheduler_step=opt.param_groups[0]['lr'])
    return np.asarray(lrs), info


def wd_only_factor(lrs, lr_scale, wd):
    """prod_k (1 - lr_scale * lr_k * wd): what decoupled AdamW decay alone (torch.optim.AdamW: p *= 1 - lr*wd)
    would leave of M over the run."""
    return float(np.exp(np.log1p(-lr_scale * wd * lrs).sum()))


# ----------------------------------------------------------------------------------------------- frequency grid
class Grid:
    def __init__(self, H, Wr):
        W = H                                    # square 224 input: W = H at every stage
        assert W // 2 + 1 == Wr, (H, Wr)
        self.H, self.W, self.Wr = H, W, Wr
        ky = np.fft.fftfreq(H)
        kx = np.fft.rfftfreq(W)
        self.ky, self.kx = ky, kx
        KY, KX = np.meshgrid(ky, kx, indexing='ij')
        self.rho = np.sqrt(KY ** 2 + KX ** 2) / 0.5
        w = np.full(Wr, 2.0)
        w[0] = 1.0
        if W % 2 == 0:
            w[-1] = 1.0
        self.w = np.broadcast_to(w, (H, Wr)).copy()
        assert abs(self.w.sum() - H * W) < 1e-9
        iy, ix = np.meshgrid(ky * H, kx * W, indexing='ij')
        self.ring = np.rint(np.sqrt(iy ** 2 + ix ** 2)).astype(int)
        self.n_ring = self.ring.max() + 1
        self.ring_rho = 2.0 * np.arange(self.n_ring) / H
        self.ring_w = np.array([self.w[self.ring == r].sum() for r in range(self.n_ring)])
        r = self.rho
        self.band = dict(DC=r < EPS, low=(r > EPS) & (r <= 0.25 + EPS), mid=(r > 0.25 + EPS) & (r <= 0.5 + EPS),
                         high=r > 0.5 + EPS)
        self.band_w = {b: float(self.w[m].sum()) for b, m in self.band.items()}

    def wmean(self, x, mask=None):
        """x (..., H, Wr) -> full-grid weighted mean over frequency (optionally inside mask); NaN for empty band."""
        w = self.w if mask is None else self.w * mask
        sw = w.sum()
        if sw == 0:
            return np.full(x.shape[:-2], np.nan)
        return (x * w).sum(axis=(-2, -1)) / sw

    def bands(self, x):
        return {b: self.wmean(x, m) for b, m in self.band.items()}

    def radial(self, x):
        """x (N, H, Wr) -> (N, n_ring) weighted ring means."""
        out = np.empty(x.shape[:-2] + (self.n_ring,))
        for r in range(self.n_ring):
            out[..., r] = self.wmean(x, self.ring == r)
        return out

    def kernel_dtft(self, K):
        """K (N, k, k) -> |K^| (N, H, Wr) on (ky, kx)."""
        k = K.shape[-1]
        u = np.arange(k)
        Ey = np.exp(-2j * np.pi * np.outer(self.ky, u))        # (H, k)
        Ex = np.exp(-2j * np.pi * np.outer(self.kx, u))        # (Wr, k)
        return np.abs(np.einsum('yu,nuv,xv->nyx', Ey, K, Ex))

    def wnorm(self, x):
        return float(np.sqrt((x ** 2 * self.w).sum()))

    def winner(self, x, y):
        return float((x * y * self.w).sum())

    def full_map(self, x):
        """half grid (H, Wr) -> fftshifted full grid (H, W) using x(-ky, -kx) = x(ky, kx)."""
        H, W = self.H, self.W
        full = np.empty((H, W))
        full[:, :self.Wr] = x
        for j in range(self.Wr, W):
            full[:, j] = x[(-np.arange(H)) % H, W - j]
        return np.fft.fftshift(full)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def stats(v):
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan, np.nan, np.nan
    return float(v.mean()), float(np.percentile(v, 25)), float(np.percentile(v, 75))


# ----------------------------------------------------------------------------------------------- analysis
def stack(d, s, h):
    blocks = sorted(b for (ss, b, hh) in d if ss == s and hh == h)
    return np.concatenate([d[(s, b, h)] for b in blocks], axis=0), blocks


BAND_STORE = {}
STORE_Q = ('psi', 'residual', 'delta_psi', 'effective_norm')


def summarize(rows, radial, key, grid, x, quantity, ratio=True):
    """band means / percentiles across (block, channel) + ring profiles of x (N, H, Wr)."""
    model, task, ckpt, s, k = key
    bm = grid.bands(x)
    if quantity in STORE_Q:
        BAND_STORE[f'{model}|{task}|s{s}|k{k}|{quantity}'] = np.stack([bm[b] for b in BANDS], 1).astype(np.float64)
    for b in BANDS:
        m, p25, p75 = stats(bm[b])
        rows.append(dict(model=model, task=task, ckpt=ckpt, stage=s, kernel=k, quantity=quantity, band=b,
                         mean=m, p25=p25, p75=p75, n=x.shape[0]))
    if ratio:
        with np.errstate(divide='ignore', invalid='ignore'):
            r_ch = bm['high'] / bm['low']
        lo, hi = np.nanmean(bm['low']) if np.isfinite(bm['low']).any() else np.nan, np.nanmean(bm['high'])
        _, p25, p75 = stats(r_ch)
        rows.append(dict(model=model, task=task, ckpt=ckpt, stage=s, kernel=k, quantity=quantity, band='high/low',
                         mean=hi / lo if np.isfinite(lo) and lo != 0 else np.nan, p25=p25, p75=p75, n=x.shape[0]))
    radial[f'{model}|{task}|s{s}|k{k}|{quantity}'] = grid.radial(x).astype(np.float32)


def analyze(args):
    model = args.model
    specs = []
    for c in args.ckpt:
        label, path = c.split('=', 1)
        specs.append((label, os.path.abspath(path)))
    assert specs[0][0] == 'IN1K', 'first --ckpt must be IN1K=<path>'
    out_dir = os.path.join(DATA, 'per_model', model)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(FIGS, exist_ok=True)

    inspect, loaded = {}, {}
    for label, path in specs:
        t0 = time.time()
        ck, linfo = load_ckpt(path)
        sd, which = select_state_dict(ck)
        filters, kernels, prefix, names = collect(sd)
        top = list(ck.keys()) if isinstance(ck, dict) else [type(ck).__name__]
        info = dict(label=label, path=os.path.relpath(path, HERE), file=os.path.basename(path),
                    size_bytes=os.path.getsize(path), sha256=sha256(path), **linfo,
                    top_level_keys=top if len(top) <= 12 else f'flat state_dict ({len(top)} tensors)',
                    state_dict_used=which, filter_key_prefix=prefix, n_filter_keys=len(names),
                    filter_key_example=names[0],
                    has_state_dict_ema=isinstance(ck, dict) and any('ema' in str(k).lower() for k in ck.keys()))
        shapes = {}
        for (s, b, h), M in filters.items():
            shapes.setdefault(s, set()).add(tuple(M.shape))
        info['filter_shapes_by_stage'] = {s: [list(x) for x in sorted(v)] for s, v in sorted(shapes.items())}
        info['blocks_by_stage'] = {s: len({b for (ss, b, hh) in filters if ss == s}) for s in sorted(shapes)}
        info['kernel_shapes_by_head'] = {h: list(kernels[min(k for k in kernels if k[2] == h)].shape)
                                         for h in sorted({k[2] for k in kernels})}
        meta = mm_meta(ck)
        if meta:
            cfg_text = meta.pop('cfg_text', None)
            info['meta'] = meta
            if cfg_text:
                with open(os.path.join(out_dir, f'cfg_{label.replace("/", "_")}.py'), 'w') as f:
                    f.write(cfg_text)
                try:
                    info['cfg'] = parse_cfg(cfg_text)
                except Exception as e:
                    info['cfg_parse_error'] = f'{type(e).__name__}: {e}'
            info['optimizer'] = optimizer_facts(ck, list(sd.keys()), names)
            ps = ck.get('param_schedulers')
            if isinstance(ps, (list, tuple)):
                info['param_schedulers_state'] = [
                    {k: v for k, v in p.items() if k in ('last_step', 'by_epoch', 'begin', 'end', 'base_values',
                                                         '_last_value', 'milestones', 'gamma', 'power', 'eta_min',
                                                         'start_factor', 'end_factor')}
                    for p in ps if isinstance(p, dict)]
        local = os.path.join(REPO, 'image_classification', 'ckpt', os.path.basename(path))
        if os.path.exists(local):
            info['local_copy'] = dict(path=os.path.relpath(local, REPO), sha256_equal=sha256(local) == info['sha256'])
        info['load_seconds'] = round(time.time() - t0, 1)
        inspect[label] = info
        loaded[label] = (filters, kernels, names)
        del ck, sd
        print(f'[{model}] loaded {label}: {info["file"]} ({info["n_filter_keys"]} filter keys, '
              f'weights_only={linfo["weights_only"]})', flush=True)

    ref_f, ref_k, ref_names = loaded['IN1K']
    for label, (f, k, names) in loaded.items():
        same = sorted(f) == sorted(ref_f) and all(f[x].shape == ref_f[x].shape for x in ref_f)
        inspect[label]['same_filter_keys_and_shapes_as_IN1K'] = bool(same)
        assert same, f'{label}: filter keys / shapes differ from IN1K'

    stages = sorted({s for (s, b, h) in ref_f})
    heads = sorted({h for (s, b, h) in ref_f})
    grids = {s: Grid(*ref_f[min(x for x in ref_f if x[0] == s)].shape[1:]) for s in stages}
    files = {label: inspect[label]['file'] for label in loaded}

    rows, radial, fits, fits_block, changes = [], {}, [], [], []
    maps = {}
    for s in stages:
        g = grids[s]
        radial[f'rho|{model}|s{s}'] = g.ring_rho
        radial[f'ringw|{model}|s{s}'] = g.ring_w
        for label in loaded:
            maps[f'map|{model}|{label}|s{s}'] = g.full_map(
                np.mean([sigmoid(stack(loaded[label][0], s, h)[0]).mean(0) for h in heads], axis=0))
        for h in heads:
            k = KSIZE[h]
            M0, blocks = stack(ref_f, s, h)
            K0, _ = stack(ref_k, s, h)
            psi0 = sigmoid(M0)
            for label, (f, kd, _) in loaded.items():
                M, _ = stack(f, s, h)
                K, _ = stack(kd, s, h)
                psi = sigmoid(M)
                key = (model, label, files[label], s, k)
                summarize(rows, radial, key, g, psi, 'psi')
                Kh = g.kernel_dtft(K)
                summarize(rows, radial, key, g, Kh, 'kernel')
                summarize(rows, radial, key, g, Kh / g.wmean(Kh)[:, None, None], 'kernel_norm')
                eff = psi * Kh
                summarize(rows, radial, key, g, eff, 'effective')
                summarize(rows, radial, key, g, eff / g.wmean(eff)[:, None, None], 'effective_norm')
                if label == 'IN1K':
                    changes.append(dict(model=model, stage=s, kernel=k, n=M0.shape[0],
                                        mean_abs_M_IN1K=float(g.wmean(np.abs(M0)).mean()),
                                        std_M_IN1K=float(np.sqrt(g.wmean((M0 - g.wmean(M0).mean()) ** 2).mean())),
                                        mean_M_IN1K=float(g.wmean(M0).mean())))
                    continue
                dM = M - M0
                summarize(rows, radial, key, g, psi - psi0, 'delta_psi', ratio=False)
                eff0 = psi0 * g.kernel_dtft(K0)
                summarize(rows, radial, key, g, eff / g.wmean(eff)[:, None, None]
                          - eff0 / g.wmean(eff0)[:, None, None], 'delta_effective_norm', ratio=False)
                a = g.winner(M, M0) / g.winner(M0, M0)
                R = M - a * M0
                summarize(rows, radial, key, g, psi - sigmoid(a * M0), 'residual', ratio=False)
                summarize(rows, radial, key, g, sigmoid(a * M0) - psi0, 'decay_part', ratio=False)
                dn = g.wnorm(dM)
                fits.append(dict(model=model, task=label, stage=s, kernel=k, a=a,
                                 rel_change=dn / g.wnorm(M0),
                                 residual_ratio=g.wnorm(R) / dn if dn > 0 else np.nan,
                                 corr=float(np.corrcoef((M * np.sqrt(g.w)).ravel(), (M0 * np.sqrt(g.w)).ravel())[0, 1]),
                                 max_abs_dM=float(np.abs(dM).max()), n=M.shape[0]))
                C4 = M.shape[0] // len(blocks)
                for j, b in enumerate(blocks):
                    Mb, M0b = M[j * C4:(j + 1) * C4], M0[j * C4:(j + 1) * C4]
                    ab = g.winner(Mb, M0b) / g.winner(M0b, M0b)
                    dnb = g.wnorm(Mb - M0b)
                    fits_block.append(dict(model=model, task=label, stage=s, block=b, kernel=k, a=ab,
                                           rel_change=dnb / g.wnorm(M0b),
                                           residual_ratio=g.wnorm(Mb - ab * M0b) / dnb if dnb > 0 else np.nan))

    # pure weight-decay reference factor per (task, stage, block) from the saved run's own schedule
    wd_ref = []
    for label in loaded:
        info = inspect[label]
        opt, cfg, meta = info.get('optimizer'), info.get('cfg'), info.get('meta')
        if not (opt and cfg and meta) or 'filters' not in opt:
            continue
        try:
            lrs, sinfo = lr_schedule(cfg, meta)
        except Exception as e:  # unknown scheduler type etc. -> no reference factor
            info['wd_only_schedule'] = dict(error=f'{type(e).__name__}: {e}')
            continue
        sinfo['adam_step_in_ckpt'] = opt['adam_steps']
        info['wd_only_schedule'] = sinfo
        for n, r in opt['filters'].items():
            m = FILTER_RE.match(n)
            wd_ref.append(dict(model=model, task=label, stage=int(m['s']) + 1, block=int(m['b']), head=int(m['h']),
                               lr_scale=r['lr_scale'], weight_decay=r['weight_decay'],
                               a_wd_only=wd_only_factor(lrs, r['lr_scale'], r['weight_decay'])))

    _write_csv(os.path.join(out_dir, 'srf_summary.csv'), rows)
    _write_csv(os.path.join(out_dir, 'fit_decay.csv'), fits)
    _write_csv(os.path.join(out_dir, 'fit_decay_by_block.csv'), fits_block)
    _write_csv(os.path.join(out_dir, 'in1k_reference.csv'), changes)
    _write_csv(os.path.join(out_dir, 'wd_only_reference.csv'), wd_ref)
    np.savez_compressed(os.path.join(out_dir, 'srf_radial.npz'), **radial)
    np.savez_compressed(os.path.join(out_dir, 'psi_maps.npz'), **maps)
    np.savez_compressed(os.path.join(out_dir, 'band_values.npz'), bands=np.array(BANDS), **BAND_STORE)
    grid_info = {s: dict(H=g.H, W=g.W, Wr=g.Wr, n_ring=int(g.n_ring), band_cells_full_grid=g.band_w)
                 for s, g in grids.items()}
    with open(os.path.join(out_dir, 'inspect.json'), 'w') as f:
        json.dump(dict(model=model, labels=list(loaded), stages=stages, heads=heads, grid=grid_info,
                       checkpoints=inspect, torch=torch.__version__, numpy=np.__version__,
                       python=sys.version.split()[0]), f, indent=1, default=str)

    import plot_srf
    plot_srf.plot_model(model, out_dir, FIGS)
    print(f'[{model}] done -> {os.path.relpath(out_dir, HERE)}', flush=True)


def _write_csv(path, rows):
    import csv
    if not rows:
        open(path, 'w').close()
        return
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f'{v:.10g}' if isinstance(v, float) else v) for k, v in r.items()})


def merge(_args):
    import csv
    models = sorted(glob.glob(os.path.join(DATA, 'per_model', '*', 'inspect.json')))
    for name in ('srf_summary.csv', 'fit_decay.csv', 'fit_decay_by_block.csv'):
        rows, header = [], None
        for m in models:
            p = os.path.join(os.path.dirname(m), name)
            with open(p) as f:
                r = list(csv.reader(f))
            if not r:
                continue
            header = header or r[0]
            assert r[0] == header
            rows += r[1:]
        with open(os.path.join(DATA, name), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
    radial = {}
    for m in models:
        with np.load(os.path.join(os.path.dirname(m), 'srf_radial.npz')) as z:
            radial.update({k: z[k] for k in z.files})
    np.savez_compressed(os.path.join(DATA, 'srf_radial.npz'), **radial)
    print('merged', [os.path.basename(os.path.dirname(m)) for m in models])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('analyze')
    a.add_argument('--model', required=True, help='e.g. S18pure, S36pure, S18hybrid, S36hybrid')
    a.add_argument('--ckpt', action='append', required=True,
                   help='LABEL=PATH; first IN1K=..., then e.g. COCO/MaskRCNN=..., ADE20K/UperNet=...')
    sub.add_parser('merge')
    args = ap.parse_args()
    sys.path.insert(0, HERE)
    {'analyze': analyze, 'merge': merge}[args.cmd](args)


if __name__ == '__main__':
    main()
