#!/bin/bash
set -e  # 오류 발생 시 스크립트 즉시 중단

echo "Initializing environment..."
unset LD_LIBRARY_PATH  # 시스템 CUDA 경로 삭제
export CUDA_DEVICE_ORDER=PCI_BUS_ID  # nvidia-smi 번호와 CUDA 번호 일치시킴

eval "$(mamba shell hook --shell bash)"
mamba activate spanetv2  # spanetv2 가상환경 활성화

echo "Running image generation training script..."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CUDA_VISIBLE_DEVICES="0,1"  # GPU IDs to use
ALL_BATCH_SIZE=256  # Target effective batch size (matches paper: total BS=256)
NUM_GPU=2
GRAD_ACCUM_STEPS=2  # RTX 4090 24GB x2: micro BS=64, accum=2 → effective BS=256

# ============================================================================
# Train DiT-S/2 on ImageNet 256×256
# Paper setting: 8 GPU, global_batch_size=256, 400K steps (~80 epochs)
# 2 GPU + grad_accum=2: micro BS=64, effective BS=256 (논문과 동일)
# ============================================================================
torchrun --nnodes=1 --master_port=29500 --nproc_per_node=$NUM_GPU "${SCRIPT_DIR}/models/dit/train.py" \
    --model DiT-S/2 \
    --data_path /dev/shm/ImageNet2012/train \
    --epochs 80 \
    --image_size 256 \
    --global_batch_size $ALL_BATCH_SIZE \
    --grad_accum_steps $GRAD_ACCUM_STEPS \
    --results_dir "${SCRIPT_DIR}/results" \
    --epoch_ckpt_every 10

# ============================================================================
# Train DiT-B/2 on ImageNet 256×256 (larger model)
# Uncomment to train:
# ============================================================================
#torchrun --nnodes=1 --master_port=29500 --nproc_per_node=$NUM_GPU "${SCRIPT_DIR}/models/dit/train.py" \
#    --model DiT-B/2 \
#    --data_path /mnt/dataset/ImageNet2012/train \
#    --epochs 80 \
#    --image_size 256 \
#    --global_batch_size $ALL_BATCH_SIZE \
#    --grad_accum_steps $GRAD_ACCUM_STEPS \
#    --results_dir "${SCRIPT_DIR}/results" \
#    --epoch_ckpt_every 10

# ============================================================================
# Train SiT-S/2 on ImageNet 256×256 (flow matching)
# Uncomment to train:
# ============================================================================
#torchrun --nnodes=1 --master_port=29500 --nproc_per_node=$NUM_GPU "${SCRIPT_DIR}/models/sit/train.py" \
#    --model SiT-S/2 \
#    --data_path /dev/shm/ImageNet2012/train \
#    --epochs 80 \
#    --image_size 256 \
#    --global_batch_size $ALL_BATCH_SIZE \
#    --grad_accum_steps $GRAD_ACCUM_STEPS \
#    --results_dir "${SCRIPT_DIR}/results" \
#    --epoch_ckpt_every 10
