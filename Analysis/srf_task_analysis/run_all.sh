#!/usr/bin/env bash
# Re-run the whole SRF analysis (CPU only, a few minutes). Checkpoints: python download_ckpts.py first.
#   bash run_all.sh                      # all four models
#   PY=/path/to/python bash run_all.sh   # python with torch + mmengine + matplotlib (default: conda env spanetv2-dense)
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-/home/server/miniforge3/envs/spanetv2-dense/bin/python}
C=ckpts
ck() { echo "$C/$1/${2%.pth}/$2"; }   # ck <tag> <file>

$PY -W ignore analyze_srf.py analyze --model S18pure \
  --ckpt IN1K=$(ck ImageNet-1K spanetv2_s18_pure_res-scale_Full-ExSPAM.pth) \
  --ckpt COCO/MaskRCNN=$(ck COCO2017 mask-rcnn_spanetv2_s18_pure_3x_epoch_35.pth) \
  --ckpt COCO/Cascade=$(ck COCO2017 cascade-mask-rcnn_spanetv2_s18_pure_3x_epoch_35.pth) \
  --ckpt ADE20K/FPN=$(ck ADE20K fpn_spanetv2_s18_pure_ade20k_467e-1.pth) \
  --ckpt ADE20K/UperNet=$(ck ADE20K upernet_spanetv2_s18_pure_ade20k_4872e-2.pth)

$PY -W ignore analyze_srf.py analyze --model S36pure \
  --ckpt IN1K=$(ck ImageNet-1K spanetv2_s36_pure_res-scale_Full-ExSPAM.pth) \
  --ckpt COCO/Cascade=$(ck COCO2017 cascade-mask-rcnn_spanetv2_s36_pure_3x_epoch_36.pth) \
  --ckpt ADE20K/FPN=$(ck ADE20K fpn_spanetv2_s36_pure_ade20k_479e-1.pth) \
  --ckpt ADE20K/UperNet=$(ck ADE20K upernet_spanetv2_s36_pure_ade20k_4976e-2.pth)

$PY -W ignore analyze_srf.py analyze --model S18hybrid \
  --ckpt IN1K=$(ck ImageNet-1K spanetv2_s18_hybrid_k7_res-scale_Full-ExSPAM.pth) \
  --ckpt COCO/MaskRCNN=$(ck COCO2017 mask-rcnn_spanetv2_s18_hybrid_3x_epoch_36.pth) \
  --ckpt COCO/Cascade=$(ck COCO2017 cascade-mask-rcnn_spanetv2_s18_hybrid_3x_epoch_35.pth) \
  --ckpt ADE20K/FPN=$(ck ADE20K fpn_spanetv2_s18_hybrid_ade20k_478e-1.pth) \
  --ckpt ADE20K/UperNet=$(ck ADE20K upernet_spanetv2_s18_hybrid_ade20k_4913e-2.pth)

$PY -W ignore analyze_srf.py analyze --model S36hybrid \
  --ckpt IN1K=$(ck ImageNet-1K spanetv2_s36_hybrid_k7_res-scale_Full-ExSPAM.pth) \
  --ckpt COCO/Cascade=$(ck COCO2017 cascade-mask-rcnn_spanetv2_s36_hybrid_3x_epoch_35.pth) \
  --ckpt ADE20K/FPN=$(ck ADE20K fpn_spanetv2_s36_hybrid_ade20k_486e-1.pth) \
  --ckpt ADE20K/UperNet=$(ck ADE20K upernet_spanetv2_s36_hybrid_ade20k_5163e-2.pth)

$PY analyze_srf.py merge
$PY make_report.py
