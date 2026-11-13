# 384 fine-tuning of spanetv2_b36_hybrid from its 224 checkpoint (meant to run on another server).
# Defaults are for 4x 24 GB GPUs; override with env vars, e.g. on 8x 80 GB GPUs:
#   DATA_PATH=/path/to/imagenet NUM_GPU=8 GRAD_ACCUM_STEPS=4 bash scripts/spanetv2/finetune_spanetv2_b36_hybrid_384.sh
# Copy ckpt/spanetv2_b36_hybrid_k7_res-scale_Full-ExSPAM.pth to the target server first (weights are not in git).

DATA_PATH=${DATA_PATH:-/home/server/dataset/ImageNet2012}
CODE_PATH=${CODE_PATH:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)} # image_classification/

# 224 checkpoint (state_dict .pth or train.py's .pth.tar), relative to CODE_PATH or absolute
INIT_CKPT=${INIT_CKPT:-./ckpt/spanetv2_b36_hybrid_k7_res-scale_Full-ExSPAM.pth}

# hybrid -> CAFormer recipe (MetaFormer scripts/caformer/finetune_caformer_b36_384.sh)
ALL_BATCH_SIZE=1024
NUM_GPU=${NUM_GPU:-4}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-32} # 4 GPUs x 8 per GPU (24 GB GPUs, estimated); per-GPU batch = 1024 / NUM_GPU / GRAD_ACCUM_STEPS
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((NUM_GPU - 1)))}  # GPU IDs to use
if (( ALL_BATCH_SIZE % (NUM_GPU * GRAD_ACCUM_STEPS) )); then
    echo "ALL_BATCH_SIZE ($ALL_BATCH_SIZE) must be divisible by NUM_GPU * GRAD_ACCUM_STEPS" >&2; exit 1
fi
let BATCH_SIZE=ALL_BATCH_SIZE/NUM_GPU/GRAD_ACCUM_STEPS

MODEL=spanetv2_b36_hybrid
IMG_SIZE=384
EPOCHS=30
OPT=lamb
LR=1e-4
SCHED=None   # constant LR, no warmup
DROP_PATH=0.8
HEAD_DROP=0.5  # applied only by train_384.py (train.py ignores it for spanetv2)
EMA_DECAY=0.9999


# train_384.py = train.py + head-dropout passed to SPANetV2
cd $CODE_PATH && torchrun --nproc_per_node=$NUM_GPU train_384.py $DATA_PATH \
--model $MODEL --img-size $IMG_SIZE --epochs $EPOCHS \
--opt $OPT --lr $LR --sched $SCHED \
-b $BATCH_SIZE --grad-accum-steps $GRAD_ACCUM_STEPS \
--initial-checkpoint $INIT_CKPT \
--mixup 0 --cutmix 0 \
--model-ema --model-ema-decay $EMA_DECAY \
--drop-path $DROP_PATH --head-dropout $HEAD_DROP --native-amp --pin-mem
