#!/usr/bin/env bash
# Full re-run (inference only, CPU). Checkpoints = the files of ../freq_robustness/weights (same release files as C-4).
set -e
cd "$(dirname "$0")"
PYC=/home/server/miniforge3/envs/spanetv2-dense/bin/python    # torch 2.1 + mmengine (needed to unpickle the mmengine checkpoints)
PYD=/home/server/miniforge3/envs/spanetv2-dense/bin/python    # pycocotools, matplotlib (images, analysis, figures)
W=../freq_robustness/weights
$PYD select_images.py
for A in pure hybrid; do
  K=$([ $A = pure ] && echo spanetv2_s18_pure_res-scale_Full-ExSPAM || echo spanetv2_s18_hybrid_k7_res-scale_Full-ExSPAM)
  MR=$([ $A = pure ] && echo mask-rcnn_spanetv2_s18_pure_3x_epoch_35 || echo mask-rcnn_spanetv2_s18_hybrid_3x_epoch_36)
  CA=cascade-mask-rcnn_spanetv2_s18_${A}_3x_epoch_35
  FP=$([ $A = pure ] && echo fpn_spanetv2_s18_pure_ade20k_467e-1 || echo fpn_spanetv2_s18_hybrid_ade20k_478e-1)
  UP=$([ $A = pure ] && echo upernet_spanetv2_s18_pure_ade20k_4872e-2 || echo upernet_spanetv2_s18_hybrid_ade20k_4913e-2)
  CUDA_VISIBLE_DEVICES= $PYC -W ignore extract.py --arch spanetv2_s18_$A \
    --ckpt IN1K=$W/cls/$K/$K.pth --ckpt COCO/MaskRCNN=$W/det/$MR/$MR.pth --ckpt COCO/Cascade=$W/det/$CA/$CA.pth \
    --ckpt ADE20K/FPN=$W/seg/$FP/$FP.pth --ckpt ADE20K/UperNet=$W/seg/$UP/$UP.pth
done
$PYD analyze.py
$PYD make_figs.py
$PYD make_report.py
