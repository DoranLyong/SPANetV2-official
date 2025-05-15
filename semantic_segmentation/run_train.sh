# -- (ref) https://mmsegmentation.readthedocs.io/en/latest/user_guides/4_train_test.html#training-and-testing-on-multiple-gpus-and-multiple-machines

#config_path=local_configs/fpn/SPANetV2/fpn_spanet_s36_pure_ade20k-40k.py
config_path=local_configs/upernet/SPANetV2/upernet_spanet_s36_pure_ade20k-160k.py
num_gpus=4 

# == Training on multiple GPUs
./tools/dist_train.sh $config_path $num_gpus


# == Training on a single GPU
#python tools/train.py  $config_path