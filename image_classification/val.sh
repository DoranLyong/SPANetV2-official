DATA_PATH=/workspace/dataset/imagenet
MODEL=spanetv2_b36_pure
CKPT=./ckpt/spanetv2_b36_pure_res-scale_Full-ExSPAM.pth

BATCH_SIZE=256

python validate.py $DATA_PATH --model $MODEL -b $BATCH_SIZE \
    --amp --pin-mem --checkpoint $CKPT
