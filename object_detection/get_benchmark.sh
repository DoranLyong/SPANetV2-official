#-- PYTHONPATH에 현재 디렉토리 추가
export PYTHONPATH=$PYTHONPATH:$(pwd)


# 설정 파일 및 체크포인트
config_path=configs/SPANetV2/cascade-mask-rcnn_spanetv2_s18_pure_fpn-3x_coco.py
ckpt_path=ckpt/spanetv2_s18_pure_res-scale_Full-ExSPAM.pth

# 벤치마크 설정
PORT=29500
REPEAT_NUM=3      # 3번 반복 측정 
MAX_ITER=2000     # 2000번 추론
LOG_INTERVAL=50   # 50번마다 로그

echo "Computing FPS benchmark for SPANetV2..."
echo "Config: $config_path"
echo "Checkpoint: $ckpt_path"
echo "Port: $PORT"

# FPS 벤치마크 실행 (추론 모드)
python tools/analysis_tools/benchmark.py \
    $config_path \
    --checkpoint $ckpt_path \
    --task inference \
    --repeat-num $REPEAT_NUM \
    --max-iter $MAX_ITER \
    --log-interval $LOG_INTERVAL \
    --num-warmup 5

echo "Benchmark completed!"