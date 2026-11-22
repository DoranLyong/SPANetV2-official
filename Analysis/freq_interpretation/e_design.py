#!/usr/bin/env python
"""E. Does the SPAM design work as intended? (ImageNet-1K weights; E1-E3 from the parameters, E4 from activations.)

    python e_design.py params      # E1, E2, E3 for S18/S36 x pure/hybrid (+ ConvFormer-S18, DFFormer-S18 for E3)
    python e_design.py acts        # E4 for S18-pure / S18-hybrid on the 1,000 images (CPU)

Frequency grid = the stored 224 grid of each stage (srf_task_analysis Grid): kernel response |K^| = DTFT of the
depthwise kernel on that grid, psi = sigmoid(M). Everything is computed on the 2D grid, never on radial averages:
  * band shares / centroids use the rfft half grid with the conjugate-symmetric weights of Grid (= full grid), DC excluded
  * Spearman correlations use every distinct frequency once (rfft half grid without the duplicate halves of the kx = 0
    and Nyquist columns), DC excluded
  power P = |K^|^2; band shares = sum_band w P / sum w P (low 0 < rho <= 0.25, mid 0.25-0.5, high > 0.5);
  centroid rho_bar = sum w rho P / sum w P.
E2 (a) absolute: Spearman(log psi_h, -log |K^_h|) per channel; (b) relative: Spearman(log psi_h,
-log(|K~_h| / mean_h' |K~_h'|)) with K~ = |K^| divided by its full-grid mean (per channel). Controls: the same
correlation with the kernel of another head of the same block and channel (3 per head) and with the kernel of a
randomly permuted channel of the same head and block (numpy default_rng(0)).
E4: per image, block and head, P2 = channel-mean power of x2 (conv output), P3 = that of x3 (SRF output); gain
log(P3 / P2) is correlated with -log P2 (absolute) and with -log(P2_h / mean_h' P2_h') (relative). Control: the gain
the same conv output would receive from another head's psi (sum_c psi'_c^2 P2_c / sum_c P2_c, same channel index).
"""
import argparse
import os
import sys

import numpy as np
import torch
from scipy.stats import rankdata

import fi_common as FC
from fi_common import C, SRF

S36 = os.path.join(FC.AN, 'srf_task_analysis', 'ckpts', 'ImageNet-1K')
WEIGHTS = {
    'spanetv2_s18_pure': lambda: C.weight_path('spanetv2_s18_pure'),
    'spanetv2_s18_hybrid': lambda: C.weight_path('spanetv2_s18_hybrid'),
    'spanetv2_s36_pure': lambda: os.path.join(S36, 'spanetv2_s36_pure_res-scale_Full-ExSPAM',
                                              'spanetv2_s36_pure_res-scale_Full-ExSPAM.pth'),
    'spanetv2_s36_hybrid': lambda: os.path.join(S36, 'spanetv2_s36_hybrid_k7_res-scale_Full-ExSPAM',
                                                'spanetv2_s36_hybrid_k7_res-scale_Full-ExSPAM.pth'),
}
KS = [3, 5, 7, 9]
BANDS = ('low', 'mid', 'high')


class G2:
    """Grid + distinct-frequency mask and band masks."""

    def __init__(self, H, Wr):
        self.g = SRF.Grid(H, Wr)
        H, W = self.g.H, self.g.W
        u = np.ones((H, Wr), bool)
        u[0, 0] = False                                            # DC
        for col in [0] + ([Wr - 1] if W % 2 == 0 else []):         # kx = 0 and Nyquist columns: keep ky in 0..H/2
            u[H // 2 + 1:, col] = False
        self.uniq = u
        self.band = {b: self.g.band[b] for b in BANDS}
        self.nodc = ~self.g.band['DC']

    def shares(self, P):
        """P (N, H, Wr) -> (N, 3) band shares and (N,) centroid, DC excluded."""
        w = self.g.w * self.nodc
        tot = (P * w).sum((-2, -1))
        sh = np.stack([(P * self.g.w * self.band[b]).sum((-2, -1)) / tot if self.band[b].any()
                       else np.full(tot.shape, np.nan) for b in BANDS], -1)          # empty band (stage 4 low) -> nan
        cen = (P * w * self.g.rho).sum((-2, -1)) / tot
        return sh, cen

    def vec(self, A):
        """(N, H, Wr) -> (N, n_distinct) values at the distinct frequencies."""
        return A[..., self.uniq]


def spearman(a, b):
    """row-wise Spearman (average ranks for ties); a, b (N, F)."""
    ra, rb = rankdata(a, axis=-1), rankdata(b, axis=-1)
    ra -= ra.mean(-1, keepdims=True)
    rb -= rb.mean(-1, keepdims=True)
    return (ra * rb).sum(-1) / np.sqrt((ra ** 2).sum(-1) * (rb ** 2).sum(-1))


def load_params(arch):
    sd = torch.load(WEIGHTS[arch](), map_location='cpu', weights_only=True)
    filters, kernels, _, _ = SRF.collect(sd)
    return filters, kernels


def params(_args):
    rows1, rows2, rows3, rows3c = [], [], [], []
    rng = np.random.default_rng(0)
    for arch in WEIGHTS:
        F_, K_ = load_params(arch)
        stages = sorted({s for (s, b, h) in F_})
        for s in stages:
            blocks = sorted({b for (ss, b, h) in F_ if ss == s})
            H, Wr = F_[(s, blocks[0], 1)].shape[1:]
            G = G2(H, Wr)
            for b in blocks:
                Kh = np.stack([G.g.kernel_dtft(K_[(s, b, h)]) for h in (1, 2, 3, 4)])          # (4, C4, H, Wr)
                psi = np.stack([SRF.sigmoid(F_[(s, b, h)]) for h in (1, 2, 3, 4)])            # (4, C4, H, Wr)
                C4 = Kh.shape[1]
                # ---- E1
                sh, cen = zip(*[G.shares(Kh[h] ** 2) for h in range(4)])
                cen = np.stack(cen)                                                          # (4, C4)
                order = np.all(np.diff(cen, axis=0) < 0, 0)                                  # k3 > k5 > k7 > k9
                pairs = np.mean([cen[i] > cen[j] for i in range(4) for j in range(i + 1, 4)], 0)
                for h in range(4):
                    for c in range(C4):
                        rows1.append((arch, s, b, KS[h], c, *[f'{x:.6g}' for x in sh[h][c]], f'{cen[h, c]:.6g}',
                                      int(order[c]), f'{pairs[c]:.4g}'))
                # ---- E2
                Kn = Kh / G.g.wmean(Kh.reshape(-1, H, Wr)).reshape(4, C4, 1, 1)
                rel = Kn / Kn.mean(0, keepdims=True)
                lpsi = np.log(G.vec(psi))                                                     # (4, C4, F)
                ka, kr = -np.log(G.vec(Kn) + 1e-30), -np.log(G.vec(rel) + 1e-30)
                perm = rng.permutation(C4)
                for h in range(4):
                    sa, sr = spearman(lpsi[h], ka[h]), spearman(lpsi[h], kr[h])
                    oa = np.stack([spearman(lpsi[h], ka[h2]) for h2 in range(4) if h2 != h])  # (3, C4)
                    orl = np.stack([spearman(lpsi[h], kr[h2]) for h2 in range(4) if h2 != h])
                    pa, pr = spearman(lpsi[h], ka[h][perm]), spearman(lpsi[h], kr[h][perm])
                    for c in range(C4):
                        rows2.append((arch, s, b, KS[h], c, f'{sa[c]:.5g}', f'{sr[c]:.5g}', f'{oa[:, c].mean():.5g}',
                                      f'{orl[:, c].mean():.5g}', f'{pa[c]:.5g}', f'{pr[c]:.5g}',
                                      f'{(sa[c] > oa[:, c]).mean():.4g}', f'{(sr[c] > orl[:, c]).mean():.4g}',
                                      int(sa[c] > pa[c]), int(sr[c] > pr[c])))
                # ---- E3
                Sk = Kh.sum(0)                                                                # (C4, H, Wr)
                Se = (psi * Kh).sum(0)
                for name, A in (('kernel_only', Sk), ('after_SRF', Se)):
                    bm = np.stack([G.g.wmean(A, G.band[bb]) for bb in BANDS], -1)            # (C4, 3)
                    cv = np.nanstd(bm, -1) / np.nanmean(bm, -1)       # stage 4 (7x7): no low band -> mid/high only
                    for c in range(C4):
                        rows3.append((arch, s, b, name, c, *[f'{x:.6g}' for x in bm[c]], f'{cv[c]:.6g}',
                                      f'{bm[c, 2] / bm[c, 0]:.6g}'))
                _, cen_k = zip(*[G.shares(Kh[h] ** 2) for h in range(4)])
                _, cen_e = zip(*[G.shares((psi[h] * Kh[h]) ** 2) for h in range(4)])
                cen_k, cen_e = np.stack(cen_k), np.stack(cen_e)
                for c in range(C4):
                    rows3c.append((arch, s, b, c, f'{np.ptp(cen_k[:, c]):.6g}', f'{np.ptp(cen_e[:, c]):.6g}',
                                   f'{cen_k[:, c].std():.6g}', f'{cen_e[:, c].std():.6g}'))
        print('done', arch, flush=True)
    # ---- E3 references: ConvFormer-S18 7x7 depthwise kernels, DFFormer-S18 predicted filters (mean over 1,000 images)
    ref = []
    sdc = torch.load(C.weight_path('convformer_s18'), map_location='cpu', weights_only=True)
    for s in (1, 2, 3, 4):
        H = 56 // 2 ** (s - 1)
        G = G2(H, H // 2 + 1)
        for k, v in sdc.items():
            if k.startswith(f'stages.{s - 1}.') and k.endswith('token_mixer.dwconv.weight'):
                A = G.g.kernel_dtft(v[:, 0].double().numpy())
                bm = np.stack([G.g.wmean(A, G.band[bb]) for bb in BANDS], -1)
                cv = np.nanstd(bm, -1) / np.nanmean(bm, -1)       # stage 4 (7x7): no low band -> mid/high only
                b = int(k.split('.')[2])
                for c in range(len(cv)):
                    ref.append(('convformer_s18', s, b, 'dwconv |K^|', c, *[f'{x:.6g}' for x in bm[c]], f'{cv[c]:.6g}',
                                f'{bm[c, 2] / bm[c, 0]:.6g}'))
    for s, b, A in dfformer_mean_filters():
        G = G2(A.shape[1], A.shape[2])
        bm = np.stack([G.g.wmean(A, G.band[bb]) for bb in BANDS], -1)
        cv = np.nanstd(bm, -1) / np.nanmean(bm, -1)       # stage 4 (7x7): no low band -> mid/high only
        for c in range(len(cv)):
            ref.append(('dfformer_s18', s, b, 'predicted |W| (image mean)', c, *[f'{x:.6g}' for x in bm[c]],
                        f'{cv[c]:.6g}', f'{bm[c, 2] / bm[c, 0]:.6g}'))
    wcsv('e1_kernel_bands.csv', ['arch', 'stage', 'block', 'kernel', 'channel', 'share_low', 'share_mid', 'share_high',
                                 'centroid_rho', 'centroids_in_kernel_order', 'pairwise_order_fraction'], rows1)
    wcsv('e2_srf_vs_kernel.csv', ['arch', 'stage', 'block', 'kernel', 'channel', 'rho_abs', 'rho_rel',
                                  'otherhead_abs_mean', 'otherhead_rel_mean', 'shuffle_abs', 'shuffle_rel',
                                  'frac_self_gt_otherhead_abs', 'frac_self_gt_otherhead_rel', 'self_gt_shuffle_abs',
                                  'self_gt_shuffle_rel'], rows2)
    wcsv('e3_flatness.csv', ['arch', 'stage', 'block', 'response', 'channel', 'band_low', 'band_mid', 'band_high',
                             'cv_across_bands', 'high_over_low'], rows3 + ref)
    wcsv('e3_centroid_spread.csv', ['arch', 'stage', 'block', 'channel', 'range_kernel', 'range_after_SRF', 'std_kernel',
                                    'std_after_SRF'], rows3c)


def dfformer_mean_filters():
    """DFFormer-S18 predicted filter magnitude |W| = |sum_f routing_f * basis_f| per channel, averaged over the 1,000
    images (routing from the reweight output via a forward hook; CPU)."""
    model = C.Classifier('dfformer_s18')
    net = model.net
    idx = FC.subset1k()
    cache = C.ValCache(1.0)
    caps, hs = {}, []
    for s, stage in enumerate(net.stages):
        for b, blk in enumerate(stage):
            tm = blk.token_mixer
            hs.append(tm.reweight.register_forward_hook(
                lambda m, i, o, key=(s + 1, b), nf=tm.num_filters: caps.setdefault(key, []).append(
                    o.detach().view(o.shape[0], nf, -1).softmax(1))))
    with torch.no_grad():
        for i in range(0, len(idx), 50):
            model(C.to01(np.array(cache.x[idx[i:i + 50]]), 'cpu'))
    for h in hs:
        h.remove()
    out = []
    for (s, b), rl in sorted(caps.items()):
        if s > 4:
            continue
        tm = net.stages[s - 1][b].token_mixer
        basis = torch.view_as_complex(tm.complex_weights.detach().contiguous())                 # (H, Wr, F)
        acc = None
        for r in rl:                                                                             # (B, F, 2C)
            Wm = torch.einsum('bfc,hwf->bchw', r.to(torch.complex64), basis).abs().sum(0)
            acc = Wm if acc is None else acc + Wm
        out.append((s, b, (acc / len(idx)).double().numpy()))
    return out


# --------------------------------------------------------------------------------------------- E4
def acts(_args):
    torch.set_num_threads(24)
    idx = FC.subset1k()
    cache = C.ValCache(1.0)
    rows = []
    for arch in ('spanetv2_s18_pure', 'spanetv2_s18_hybrid'):
        model = C.Classifier(arch)
        net = model.net
        grids = {}
        caps, hs = {}, []
        for s, stage in enumerate(net.stages):
            for b, blk in enumerate(stage):
                tm = blk.token_mixer
                if type(tm).__name__ != 'ExSPAM':
                    continue
                for h in range(tm.num_heads):
                    hs.append(getattr(tm, f'local_conv_{h + 1}').register_forward_hook(
                        lambda m, i, o, key=(s + 1, b, h): caps.__setitem__((*key, 'x2'), o.detach())))
                hs.append(tm.proj.register_forward_pre_hook(
                    lambda m, i, key=(s + 1, b), nh=tm.num_heads: caps.__setitem__((*key, 's_in'), i[0].detach())))
        for i0 in range(0, len(idx), 50):
            caps.clear()
            with torch.no_grad():
                model(C.to01(np.array(cache.x[idx[i0:i0 + 50]]), 'cpu'))
            blocks = sorted({k[:2] for k in caps if k[-1] == 's_in'})
            for (s, b) in blocks:
                sin = caps[(s, b, 's_in')]
                B, Cc, H, W = sin.shape
                v = sin.view(B, Cc // 4, 4, H, W)
                if (H, W) not in grids:
                    grids[(H, W)] = G2(H, W // 2 + 1)
                G = grids[(H, W)]
                P2c = [(torch.fft.rfft2(caps[(s, b, h, 'x2')].double(), norm='ortho').abs() ** 2).numpy()
                       for h in range(4)]                                                     # 4 x (B, C4, H, Wr)
                P2 = np.stack([p.mean(1) for p in P2c])                                       # (4, B, H, Wr)
                tm = net.stages[s - 1][b].token_mixer
                psi2 = [torch.sigmoid(getattr(tm, f'filter_{h + 1}')).detach().double().numpy() ** 2 for h in range(4)]
                P3 = np.stack([(torch.fft.rfft2(v[:, :, h].double(), norm='ortho').abs() ** 2).mean(1).numpy()
                               for h in range(4)])
                gain = np.log(G.vec(P3) / G.vec(P2))
                P2n = P2 / np.stack([G.g.wmean(P2[h]) for h in range(4)])[:, :, None, None]
                relw = -np.log(G.vec(P2n / P2n.mean(0, keepdims=True)))
                absw = -np.log(G.vec(P2n))
                sh2 = np.stack([G.shares(P2[h])[0] for h in range(4)])                            # (4, B, 3)
                sh3 = np.stack([G.shares(P3[h])[0] for h in range(4)])
                rel_sh2 = sh2 / sh2.mean(0, keepdims=True)
                for h in range(4):
                    ra, rr = spearman(gain[h], absw[h]), spearman(gain[h], relw[h])
                    # control: the gain this head's conv output would get from another head's psi (same channel index)
                    ca, cr = [], []
                    for h2 in range(4):
                        if h2 == h:
                            continue
                        gc = np.log(G.vec((psi2[h2][None] * P2c[h]).mean(1)) / G.vec(P2[h]))
                        ca.append(spearman(gc, absw[h]))
                        cr.append(spearman(gc, relw[h]))
                    ca, cr = np.stack(ca), np.stack(cr)                                       # (3, B)
                    wa, wr = np.nanargmin(sh2[h], -1), np.nanargmin(rel_sh2[h], -1)
                    up_a = sh3[h][np.arange(B), wa] > sh2[h][np.arange(B), wa]
                    up_r = sh3[h][np.arange(B), wr] > sh2[h][np.arange(B), wr]
                    for j in range(B):
                        rows.append((arch, s, b, KS[h], int(idx[i0 + j]), f'{ra[j]:.5g}', f'{rr[j]:.5g}',
                                     f'{ca[:, j].mean():.5g}', f'{cr[:, j].mean():.5g}', f'{(ra[j] > ca[:, j]).mean():.4g}',
                                     f'{(rr[j] > cr[:, j]).mean():.4g}', BANDS[wa[j]], int(up_a[j]), BANDS[wr[j]],
                                     int(up_r[j])))
            print(arch, i0 + 50, flush=True)
        for h_ in hs:
            h_.remove()
    wcsv('e4_activation_gain.csv', ['arch', 'stage', 'block', 'kernel', 'image', 'rho_abs', 'rho_rel',
                                    'otherhead_psi_rho_abs', 'otherhead_psi_rho_rel', 'frac_self_gt_otherhead_abs',
                                    'frac_self_gt_otherhead_rel', 'weakest_band_abs',
                                    'weak_band_share_up_abs', 'weakest_band_rel', 'weak_band_share_up_rel'], rows)


def wcsv(name, header, rows):
    import csv
    with open(os.path.join(FC.DATA, name), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print('wrote', name, len(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('part', choices=['params', 'acts'])
    args = ap.parse_args()
    os.makedirs(FC.DATA, exist_ok=True)
    {'params': params, 'acts': acts}[args.part](args)


if __name__ == '__main__':
    main()
