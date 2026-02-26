# -- (ref) https://mmsegmentation.readthedocs.io/en/latest/user_guides/4_train_test.html#training-and-testing-on-multiple-gpus-and-multiple-machines


#-- PYTHONPATH에 현재 디렉토리 추가
export PYTHONPATH=$PYTHONPATH:$(pwd)

#config_path=local_configs/fpn/SPANetV2/fpn_spanet_s36_pure_ade20k-40k.py
#config_path=local_configs/upernet/SPANetV2/upernet_spanet_s36_pure_ade20k-160k.py


config_path=local_configs/upernet/Metaformer/upernet_caformer_s36_ade20k-160k.py


num_gpus=4 

# == Training on multiple GPUs
#CUDA_VISIBLE_DEVICES=5,6 ./tools/dist_train.sh $config_path $num_gpus
#CUDA_VISIBLE_DEVICES=5,6 ./tools/dist_train.sh $config_path $num_gpus --resume


# == Training on a single GPU
CUDA_VISIBLE_DEVICES=0 python tools/train.py  $config_path