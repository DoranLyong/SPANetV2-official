DATA_PATH=/workspace/dataset/imagenet
CODE_PATH=/workspace/projects/SPANetV2_proj/SPANetV2_on_MetaFormer_ver2 # modify code path here

export CUDA_VISIBLE_DEVICES="0,1,2,3"  # GPU IDs to use

ALL_BATCH_SIZE=4096  # 4 * 1024
NUM_GPU=4
GRAD_ACCUM_STEPS=8 # Adjust according to your GPU numbers and memory size.
let BATCH_SIZE=ALL_BATCH_SIZE/NUM_GPU/GRAD_ACCUM_STEPS

MODEL=spanetv2_s18_pure
OPT=adamw # {adamw, lamb}
LR=4e-3  # 4 * 1e-3
WARMUP_EPOCH=20  # 4 * 5
DROP_PATH=0.2
HEAD_DROP=0.0


cd $CODE_PATH && sh distributed_train.sh $NUM_GPU $DATA_PATH \
--model $MODEL --opt $OPT --lr $LR --warmup-epochs $WARMUP_EPOCH \
-b $BATCH_SIZE --grad-accum-steps $GRAD_ACCUM_STEPS \
--drop-path $DROP_PATH --head-dropout $HEAD_DROP --native-amp --pin-mem \
#--resume ./output/train/20251126-154210-spanetv2_s16_pure-224/checkpoint-26.pth.tar
#--clip-mode norm
