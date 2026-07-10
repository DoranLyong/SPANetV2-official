# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
import os
import wandb
import logging
import argparse
import numpy as np
from PIL import Image
from glob import glob
from time import time
from copy import deepcopy
from collections import OrderedDict

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchvision.datasets as datasets
from torchvision import transforms
from torchvision.datasets import ImageFolder

from models import DiT_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL

# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.Resampling.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.Resampling.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def estimated_time(t_start, cur_iter, start_iter, total_iter):
    t_curr = time()
    eta_total = (t_curr - t_start) / (cur_iter + 1 - start_iter) * (total_iter - cur_iter - 1)
    eta_hour = int(eta_total // 3600)
    eta_min = int((eta_total - eta_hour * 3600) // 60)
    eta_sec = int(eta_total - eta_hour * 3600 - eta_min * 60)
    return f'{eta_hour:02d} h {eta_min:02d} m {eta_sec:02d} s', (eta_total / 3600.)


def find_latest_checkpoint(checkpoint_dir):
    """
    Find the latest checkpoint in the checkpoint directory.
    Returns the path to the latest checkpoint, or None if no checkpoint exists.
    """
    if not os.path.exists(checkpoint_dir):
        return None
    
    # Find all epoch-based checkpoints
    epoch_ckpts = glob(f"{checkpoint_dir}/epoch_*.pt")
    if epoch_ckpts:
        # Sort by modification time and return the latest
        latest_ckpt = max(epoch_ckpts, key=os.path.getmtime)
        return latest_ckpt
    
    return None


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    grad_accum_steps = args.grad_accum_steps
    assert args.global_batch_size % (dist.get_world_size() * grad_accum_steps) == 0, \
        f"Batch size must be divisible by world_size * grad_accum_steps."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # wandb
    if args.use_wandb and dist.get_rank() == 0:
        wandb.login(key="3ce6822247e17e443d7597881467081571a6c0f4")  # puyifan
        wandb.init(
            config=args,
            project="Linear_DiT",
            entity="yifanpu001",
            name=args.wandb_name,
            dir=args.results_dir,
            reinit=True,
        )

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        # experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        # experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        experiment_dir = f"{args.results_dir}/{model_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        experiment_dir = f"{args.results_dir}/{model_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        logger = create_logger(experiment_dir)

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
    )
    # Note that parameter initialization is done within the DiT constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    model = DDP(model.to(device), device_ids=[rank], find_unused_parameters=True)
    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

    # Setup data:
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    if args.dummy:
        logger.info(f"[WARNING] You are using dummy data, which is only used for debugging!")
        dataset = datasets.FakeData(1281167, (3, args.image_size, args.image_size), 1000, transforms.ToTensor())
    else:
        dataset = ImageFolder(args.data_path, transform=transform)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    micro_batch_size = int(args.global_batch_size // (dist.get_world_size() * grad_accum_steps))
    loader = DataLoader(
        dataset,
        batch_size=micro_batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")
    logger.info(f"Gradient accumulation steps: {grad_accum_steps}")
    logger.info(f"Micro batch size per GPU: {micro_batch_size}")
    logger.info(f"Effective batch size: {args.global_batch_size}")

    # Prepare models for training:
    update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    resume_start_step = 0
    start_time = time()
    global_start_time = time()

    # auto resume is default True
    if True:
        latest_ckpt = find_latest_checkpoint(checkpoint_dir)
        if latest_ckpt is not None:
            args.resume_ckpt = latest_ckpt
            logger.info(f"Auto resume enabled: found checkpoint {latest_ckpt}")
        else:
            logger.info("Auto resume enabled: no checkpoint found, starting from scratch")

    # Resume
    if (args.resume_ckpt is not None) and (os.path.exists(args.resume_ckpt)):
        logger.info(f'Start resume from {args.resume_ckpt}')
        ckpt = torch.load(args.resume_ckpt, map_location='cpu')
        args.start_epoch = ckpt['epoch']
        model.module.load_state_dict(ckpt['model'])
        ema.load_state_dict(ckpt['ema'])
        opt.load_state_dict(ckpt['opt'])
        train_steps = ckpt['train_steps']
        resume_start_step = train_steps
        logger.info(f'Finish resume, from finished epoch {args.start_epoch}.')

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        opt.zero_grad()
        for micro_step, (x, y) in enumerate(loader):
            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = dict(y=y)
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            loss = loss_dict["loss"].mean() / grad_accum_steps
            loss.backward()

            # Optimizer step on accumulation boundary
            if (micro_step + 1) % grad_accum_steps == 0:
                opt.step()
                opt.zero_grad()
                update_ema(ema, model.module)
                train_steps += 1

            # Log loss values:
            running_loss += loss.item() * grad_accum_steps  # recover unscaled loss for logging
            log_steps += 1
            if train_steps > 0 and train_steps % args.log_every == 0 and (micro_step + 1) % grad_accum_steps == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                total_opt_steps = args.epochs * len(loader) // grad_accum_steps
                logger.info(f"(step={train_steps:07d}/{total_opt_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}, Elapsed Time: {((time() - global_start_time) / 3600.):.2f} h, ETA: {estimated_time(global_start_time, train_steps, resume_start_step, total_opt_steps)[0]}")
                # wandb
                if args.use_wandb and dist.get_rank() == 0:
                    wandb.log(
                        {"train_loss": avg_loss, "train_steps_per_sec": steps_per_sec, "used_hours": (time() - global_start_time) / 3600.,
                        "remain_hours": estimated_time(global_start_time, train_steps, resume_start_step, total_opt_steps)[1]},
                        step=train_steps)
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save DiT checkpoint (only on accumulation boundary):
            if (micro_step + 1) % grad_accum_steps == 0 and train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()

        # Save DiT checkpoint, epoch-based, can be resume:
        if (epoch + 1) % args.epoch_ckpt_every == 0:
            if rank == 0:
                checkpoint = {
                    'epoch': epoch + 1,
                    "model": model.module.state_dict(),
                    "ema": ema.state_dict(),
                    "opt": opt.state_dict(),
                    "train_steps": train_steps,
                    "args": args
                }
                checkpoint_path = f"{checkpoint_dir}/epoch_{epoch:03d}.pt"
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
            dist.barrier()

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image_size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num_classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global_batch_size", type=int, default=256)
    parser.add_argument("--global_seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--ckpt_every", type=int, default=50_000)
    parser.add_argument("--epoch_ckpt_every", type=int, default=1)
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--resume_ckpt", type=str, help="ckpt to resume")
    parser.add_argument('--dummy', action='store_true', help="use fake data to benchmark")
    ## wandb
    parser.add_argument("--use_wandb", action="store_true", default=False)
    parser.add_argument("--wandb_name", type=str)
    parser.add_argument("--grad_accum_steps", type=int, default=1,
                        help="gradient accumulation steps (default: 1)")
    args = parser.parse_args()

    main(args)
