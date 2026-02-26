#-- PYTHONPATH에 현재 디렉토리 추가
export PYTHONPATH=$PYTHONPATH:$(pwd)


#config_path=local_configs/fpn/SPANetV2/fpn_spanet_s18_pure_ade20k-40k.py
#ckpt_path=results/spanetv2/fpn_spanetv2_s18_pure_ade20k_467e-1.pth

config_path=configs/poolformer/fpn_poolformer_s24_8xb4-40k_ade20k-512x512.py
ckpt_path=results/poolformer/fpn_poolformer_s24.pth

#config_path=local_configs/upernet/SPANetV2/upernet_spanet_s18_pure_ade20k-160k.py
#ckpt_path=results/upernet/upernet_spanetv2_s18_pure_ade20k_4872e-2.pth


python tools/analysis_tools/benchmark.py \
        $config_path \
        $ckpt_path