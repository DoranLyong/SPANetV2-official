# (ref) https://mmsegmentation.readthedocs.io/en/latest/user_guides/4_train_test.html#testing-and-saving-segment-files

config_path=local_configs/upernet/SPANetV2/upernet_spanet_s36_hybrid_ade20k-160k.py
ckpt_path=results/upernet_spanetv2_s36_hybrid_ade20k_5163e-2.pth

#config_path=local_configs/fpn/SPANetV2/fpn_spanet_s36_pure_ade20k-40k.py
#ckpt_path=results/fpn_spanetv2_s36_pure_ade20k_479e-1.pth


# == Test on a single GPU 
python tools/test.py  $config_path $ckpt_path 
