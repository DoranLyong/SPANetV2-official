DATA_PATH=/workspace/dataset/imagenet
CODE_PATH=/workspace/projects/SPANetV2_on_MetaFormer_ver2 # modify code path here


ALL_BATCH_SIZE=4096  # 4 * 1024
NUM_GPU=4
GRAD_ACCUM_STEPS=16 # Adjust according to your GPU numbers and memory size.
let BATCH_SIZE=ALL_BATCH_SIZE/NUM_GPU/GRAD_ACCUM_STEPS

MODEL=spanetv2_s36_pure
OPT=adamw # {adamw, lamb}
LR=4e-3  # 4 * 1e-3
WARMUP_EPOCH=20  # 4 * 5
DROP_PATH=0.3
HEAD_DROP=0.0



cd $CODE_PATH && sh distributed_train.sh $NUM_GPU $DATA_PATH \
--model $MODEL --opt $OPT --lr $LR --warmup-epochs $WARMUP_EPOCH \
-b $BATCH_SIZE --grad-accum-steps $GRAD_ACCUM_STEPS \
--drop-path $DROP_PATH --head-dropout $HEAD_DROP --native-amp --pin-mem \
