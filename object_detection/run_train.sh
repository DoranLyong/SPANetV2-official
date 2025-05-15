# -- (ref) https://mmsegmentation.readthedocs.io/en/latest/user_guides/4_train_test.html#training-and-testing-on-multiple-gpus-and-multiple-machines

config_path=local_configs/SPANetV2/cascade-mask-rcnn_spanetv2_s18_hybrid_fpn-3x_coco.py
#config_path=local_configs/SPANetV2/mask-rcnn_spanetv2_s18_pure_3x.py
num_gpus=4

# == Training on multiple GPUs
./tools/dist_train.sh $config_path $num_gpus


# == Training on a single GPU
#python tools/train.py  $config_path