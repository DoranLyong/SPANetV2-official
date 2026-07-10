"""
ExSPAM DiT/SiT model analysis: Params, FLOPs, Inference speed
"""
import sys
import os
import torch
import time
import argparse
from fvcore.nn import FlopCountAnalysis, flop_count_table


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='DiT-S/2',
                        help='Model name (e.g., DiT-S/2, DiT-S/4, DiT-S/8, DiT-B/2)')
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--framework', type=str, default='dit', choices=['dit', 'sit'],
                        help='dit or sit')
    args = parser.parse_args()

    # Import model based on framework
    if args.framework == 'dit':
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models', 'dit'))
        from models import DiT_models
    else:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models', 'sit'))
        from models import SiT_models as DiT_models

    # Build model
    model_name = args.model
    patch_size = int(model_name.split('/')[-1])
    latent_size = args.image_size // 8  # VAE downsamples 8x
    input_size = latent_size // patch_size * patch_size  # ensure divisible

    model = DiT_models[model_name](input_size=latent_size, num_classes=1000)
    model.eval()

    info = []

    # ---------- Parameters ----------
    info.append(f'Model:\t{model_name} ({args.framework.upper()})')
    info.append(f'Params:\t{count_parameters(model) / 1e6:.2f} M')

    # ---------- FLOPs ----------
    x = torch.randn(1, 4, latent_size, latent_size)  # VAE latent (4 channels)
    t = torch.tensor([0])
    y = torch.tensor([0])  # class label
    flops = FlopCountAnalysis(model, (x, t, y))
    info.append(f'FLOPs:\t{flops.total() / 1e9:.2f} G')
    print(flop_count_table(flops, max_depth=2))

    # ---------- Inference Speed (CPU) ----------
    n = 10
    x_batch = torch.randn(n, 4, latent_size, latent_size)
    t_batch = torch.zeros(n, dtype=torch.long)
    y_batch = torch.zeros(n, dtype=torch.long)
    with torch.no_grad():
        # warmup
        model(x_batch[:1], t_batch[:1], y_batch[:1])
        torch.cpu.synchronize()
        start = time.time()
        model(x_batch, t_batch, y_batch)
        torch.cpu.synchronize()
        end = time.time()
    info.append(f'Time:\t{(end - start) * 1000 / n:.2f} ms/image (CPU)')

    print('\n')
    for item in info:
        print(item)


if __name__ == '__main__':
    main()
