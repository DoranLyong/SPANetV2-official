#config_path=local_configs/fpn/SPANetV2/fpn_spanet_s36_pure_ade20k-40k.py
#ckpt_path=results/upernet_spanetv2_s18_pure_ade20k_4823e-2.pth

config_path=local_configs/upernet/SPANetV2/upernet_spanet_s18_pure_ade20k-160k.py
ckpt_path=results/upernet/upernet_spanetv2_s18_pure_ade20k_4872e-2.pth


python tools/analysis_tools/benchmark.py $config_path $ckpt_path