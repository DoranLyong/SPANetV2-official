# (ref) https://mmsegmentation.readthedocs.io/en/latest/user_guides/4_train_test.html#testing-and-saving-segment-files

#-- PYTHONPATH에 현재 디렉토리 추가
export PYTHONPATH=$PYTHONPATH:$(pwd)

config_path=local_configs/fpn/SPANetV2/fpn_spanet_s18_pure_ade20k-40k.py
ckpt_path=results/spanetv2/fpn_spanetv2_s18_pure_ade20k_467e-1.pth

#config_path=local_configs/fpn/SPANetV2/fpn_spanet_s36_pure_ade20k-40k.py
#ckpt_path=results/fpn_spanetv2_s36_pure_ade20k_479e-1.pth


# == Test on a single GPU 
CUDA_VISIBLE_DEVICES=0 python tools/test.py  $config_path $ckpt_path 
