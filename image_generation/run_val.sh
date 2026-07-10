#!/bin/bash
set -e

echo "Initializing environment..."
unset LD_LIBRARY_PATH
export CUDA_DEVICE_ORDER=PCI_BUS_ID

eval "$(mamba shell hook --shell bash)"
mamba activate spanetv2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CUDA_VISIBLE_DEVICES="0,1"
NUM_GPU=2

# ============================================================================
# Configuration (환겨에 맞게 수정 필요)
# ============================================================================
CKPT="${SCRIPT_DIR}/results/DiT-S-2/checkpoints/epoch_079.pt"
REF_NPZ="${SCRIPT_DIR}/tools/VIRTUAL_imagenet256_labeled.npz"


# ============================================================================
# Step 1: Download ADM official reference npz (최초 1회만, 이미 있으면 스킵)
# Source: https://github.com/openai/guided-diffusion/tree/main/evaluations
# ============================================================================
if [ ! -f "$REF_NPZ" ]; then
    echo "Downloading ADM official reference npz..."
    wget -O "$REF_NPZ" \
        "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz"
fi

# -- (Alternative) Create reference npz from local ImageNet val:
#python "${SCRIPT_DIR}/tools/create_ref_npz.py" \
#    --ref_folder /dev/shm/ImageNet2012/val \
#    --output "$REF_NPZ" --num_workers 16



# ============================================================================
# Step 2: Generate 50K images (PNG) from the trained model (DiT-S/2)
# ============================================================================
torchrun --nnodes=1 --master_port=29500 --nproc_per_node=$NUM_GPU "${SCRIPT_DIR}/models/dit/sample_ddp_nonpz.py" \
    --model DiT-S/2 \
    --num_fid_samples 50000 \
    --image_size 256 \
    --cfg_scale 1.0 \
    --per_proc_batch_size 64 \
    --ckpt "$CKPT" \
    --sample_dir "${SCRIPT_DIR}/results"


# ============================================================================
# Step 2: Generate 50K images (PNG) from the trained model (SiT-S/2)
# ============================================================================
#torchrun --nnodes=1 --master_port=29500 --nproc_per_node=$NUM_GPU "${SCRIPT_DIR}/models/sit/sample_ddp_nonpz.py" ODE \
#    --model SiT-S/2 \
#    --num_fid_samples 50000 \
#    --image_size 256 \
#    --cfg_scale 1.0 \
#    --per_proc_batch_size 64 \
#    --ckpt "$CKPT" \
#    --sample_dir "${SCRIPT_DIR}/results"


# ============================================================================
# Step 2.5: Convert the generate PNG to npz
# ============================================================================
SAMPLE_FOLDER=$(cat "${SCRIPT_DIR}/results/folder_name.txt")
python "${SCRIPT_DIR}/tools/create_ref_npz.py" \
    --ref_folder "${SCRIPT_DIR}/results/${SAMPLE_FOLDER}" \
    --output "${SCRIPT_DIR}/results/${SAMPLE_FOLDER}.npz" \
    --num 50000 --image_size 256 --num_workers 16


# ============================================================================
# Step 3: Compute FID between Step1 and Step 2.5 via tensorflow
# ============================================================================
# arg1: reference npz (ImageNet val from Step 1)
# arg2: results dir (generated npz from Step 2.5 is auto-detected via folder_name.txt)

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

CUDA_VISIBLE_DEVICES="0" python "${SCRIPT_DIR}/tools/evaluator_read.py" \
    "$REF_NPZ" \
    "${SCRIPT_DIR}/results" \
    --batch_size 256