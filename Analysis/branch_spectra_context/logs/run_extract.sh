set -e
cd /home/server/Workspace/SPANetV2-official/Analysis/branch_spectra_context
PYC=/home/server/miniforge3/envs/spanetv2-dense/bin/python
W=../freq_robustness/weights
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
