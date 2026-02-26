# (ref) https://mmsegmentation.readthedocs.io/en/latest/user_guides/4_train_test.html#testing-and-saving-segment-files

#-- PYTHONPATH에 현재 디렉토리 추가
export PYTHONPATH=$PYTHONPATH:$(pwd)

#config_path=local_configs/SPANetV2/mask-rcnn_spanetv2_s18_pure_3x.py
#ckpt_path=results/mask-rcnn_spanetv2_s18_pure_3x_epoch_35.pth

config_path=configs/SPANetV2/cascade-mask-rcnn_spanetv2_s18_pure_fpn-3x_coco.py
ckpt_path=results/cascade-mask-rcnn_spanetv2_s18_pure_3x_epoch_35.pth


# == Test on a single GPU 
CUDA_VISIBLE_DEVICES=0 python tools/test.py  $config_path $ckpt_path --eval bbox segm 
