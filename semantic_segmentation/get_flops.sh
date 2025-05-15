#config_path=local_configs/fpn/SPANetV2/fpn_spanet_s18_hybrid_ade20k-40k.py
config_path=local_configs/upernet/SPANetV2/upernet_spanet_s36_pure_ade20k-160k.py
#config_path=configs/convnext/convnext-tiny_upernet_8xb2-amp-160k_ade20k-512x512.py


python tools/analysis_tools/get_flops.py $config_path --shape 2048 512