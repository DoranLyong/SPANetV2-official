#!/usr/bin/env bash
# D: ADE20K (seg_eval.py) then COCO (det_eval.py) inference on one GPU; waits for each weight file to be downloaded.
#   bash run_dense.sh <gpu>
cd "$(dirname "$0")"
PY=${PY:-/home/server/miniforge3/envs/spanetv2-dense/bin/python}
gpu=${1:-3}
wait_for() {  # wait until the weight file of $1 exists
  for i in $(seq 1 720); do
    p=$($PY -c "import sys; sys.argv=['x']; import importlib.util as u; s=u.spec_from_file_location('m','$2'); m=u.module_from_spec(s); s.loader.exec_module(m); print(m.weight_path('$1'))" 2>/dev/null)
    [ -n "$p" ] && [ -f "$p" ] && return 0; sleep 30
  done; return 1
}
SEG="upernet_spanetv2_s18_pure upernet_spanetv2_s18_hybrid upernet_spanetv2_s36_pure upernet_spanetv2_s36_hybrid
     fpn_spanetv2_s18_pure fpn_spanetv2_s18_hybrid fpn_spanetv2_s36_pure fpn_spanetv2_s36_hybrid
     upernet_convnext_tiny upernet_convnext_small upernet_swin_tiny upernet_swin_small fpn_poolformer_s24 fpn_poolformer_s36"
DET="mask_rcnn_spanetv2_s18_pure mask_rcnn_spanetv2_s18_hybrid cascade_spanetv2_s18_pure cascade_spanetv2_s18_hybrid
     cascade_spanetv2_s36_pure cascade_spanetv2_s36_hybrid mask_rcnn_convnext_tiny cascade_convnext_tiny
     cascade_convnext_small mask_rcnn_swin_tiny cascade_swin_tiny cascade_swin_small"
for m in $SEG; do
  [ -f data/raw/seg_$m.json ] && { echo "skip seg $m"; continue; }
  wait_for $m seg_eval.py || { echo "no weight $m"; continue; }
  echo "[$(date '+%F %T')] seg $m"
  CUDA_VISIBLE_DEVICES=$gpu $PY -W ignore seg_eval.py --model $m > logs/seg_$m.log 2>&1; echo "  exit=$?"
done
for m in $DET; do
  [ -f data/raw/det_$m.json ] && { echo "skip det $m"; continue; }
  wait_for $m det_eval.py || { echo "no weight $m"; continue; }
  echo "[$(date '+%F %T')] det $m"
  CUDA_VISIBLE_DEVICES=$gpu $PY -W ignore det_eval.py --model $m > logs/det_$m.log 2>&1; echo "  exit=$?"
done
echo "[$(date '+%F %T')] DENSE DONE"
