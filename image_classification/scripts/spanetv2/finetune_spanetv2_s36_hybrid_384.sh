DATA_PATH=/home/server/dataset/ImageNet2012
CODE_PATH=/home/server/Workspace/SPANetV2-official/image_classification # modify code path here

# 224 checkpoint (state_dict .pth or train.py's .pth.tar), relative to CODE_PATH or absolute
INIT_CKPT=./ckpt/spanetv2_s36_hybrid_k7_res-scale_Full-ExSPAM.pth

export CUDA_VISIBLE_DEVICES="0,1,2,3"  # GPU IDs to use

# hybrid -> CAFormer recipe (MetaFormer scripts/caformer/finetune_caformer_s36_384.sh)
ALL_BATCH_SIZE=1024
NUM_GPU=4
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-16} # 16 per GPU, ~19 GiB on RTX 4090 (32 per GPU: OOM)
let BATCH_SIZE=ALL_BATCH_SIZE/NUM_GPU/GRAD_ACCUM_STEPS

MODEL=spanetv2_s36_hybrid
IMG_SIZE=384
EPOCHS=30
OPT=lamb
LR=1e-4
SCHED=None   # constant LR, no warmup
DROP_PATH=0.5
HEAD_DROP=0.4  # applied only by train_384.py (train.py ignores it for spanetv2)
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
