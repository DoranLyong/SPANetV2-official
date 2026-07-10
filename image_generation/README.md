# Image Generation via SPAM

SPAM block을 DiT/SiT에 적용하여 image generation 성능을 평가.
ViTTT 논문(DiT³)의 TTT block 결과와 비교.

## Training Setup

| Item | Value |
|------|-------|
| Dataset | ImageNet-1K 256x256 |
| Optimizer | AdamW (lr=1e-4 constant, wd=0) |
| Batch size | 256 |
| Training | 80 epochs (~400K steps) |
| EMA decay | 0.9999 |
| VAE | stabilityai/sd-vae-ft-ema |

## Results: DiT-S/2

FID-50K evaluated with ADM official reference (`VIRTUAL_imagenet256_labeled.npz`), cfg_scale=1.0 (no classifier-free guidance), 250 DDPM sampling steps.

| Method | Token Mixer | Params | FLOPs | FID↓ | sFID↓ | IS↑ | Precision↑ | Recall↑ | Note |
|--------|------------|--------|-------|------|-------|-----|-----------|---------|------|
| DiT-S/2 (논문) | Self-Attention | 33M | 6.06G | 68.40 | - | - | - | - | baseline |
| DiT³-S/2 (논문) | TTT | 35M | 6.23G | 62.65 | - | 21.59 | 0.39 | 0.57 | 80 epoch |
| **DiT-S/2 + SPAM** | **SPAM** | **35.7M** | **6.03G** | **61.72** | **12.39** | **23.62** | **0.428** | **0.549** | **80 epoch** |


## Results: SiT-S/2

FID-50K evaluated with ADM official reference (`VIRTUAL_imagenet256_labeled.npz`), cfg_scale=1.0, ODE sampling (dopri5, 250 steps).

| Method | Token Mixer | Params | FLOPs | FID↓ | sFID↓ | IS↑ | Precision↑ | Recall↑ | Note |
|--------|------------|--------|-------|------|-------|-----|-----------|---------|------|
| SiT-S/2 (논문) | Self-Attention | 33M | 6.1G | 57.3 | - | - | - | - | baseline |
| SiT-S/2 + VCA (논문) | VCA | 33.6M | 6.0G | 53.0 | - | - | - | - | LinearDiff |
| **SiT-S/2 + SPAM** | **SPAM** | **35.7M** | **6.0G** | **54.98** | **9.73** | **26.79** | **0.462** | **0.574** | **80 epoch** |

## Notes

- **DiT-S/2**: SPAM FID 61.72 → DiT³(TTT) 62.65, VCA 62.3 대비 모두 개선 (**best**)
- **SiT-S/2**: SPAM FID 54.98 → vanilla 57.3 대비 -2.32 개선, VCA 53.0 대비 +1.98
- FID가 epoch 70→80에서 63.98→61.72로 여전히 개선 중 (saturate하지 않음)
- 평가 기준: ADM 공식 reference npz, FID-50K, cfg_scale=1.0
- DiT: 250 DDPM sampling steps, SiT: ODE sampling (dopri5, 250 steps)
- SPAM은 Self-Attention을 frequency-domain filtering + multi-scale conv로 교체한 모듈
