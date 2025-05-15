# (ref) https://mmsegmentation.readthedocs.io/en/latest/user_guides/4_train_test.html#testing-and-saving-segment-files

#config_path=local_configs/SPANetV2/mask-rcnn_spanetv2_s18_pure_3x.py
#ckpt_path=results/mask-rcnn_spanetv2_s18_pure_3x_epoch_35.pth

config_path=local_configs/SPANetV2/cascade-mask-rcnn_spanetv2_s18_hybrid_fpn-3x_coco.py
ckpt_path=results/cascade-mask-rcnn_spanetv2_s18_hybrid_3x_epoch_35.pth

num_gpus=1

#./tools/dist_train.sh $config_path $num_gpus

# == Test on a single GPU 
python tools/test.py  $config_path $ckpt_path 
