# Constructed for the re-measurement (not in any repository):
# SPANetV2-S18-pure Cascade Mask R-CNN 3x config with only the backbone replaced (neck, heads, test_cfg unchanged).
_base_ = '../../../../object_detection/local_configs/SPANetV2/cascade-mask-rcnn_spanetv2_s18_pure_fpn-3x_coco.py'
model = dict(backbone=dict(_delete_=True, type='convformer_s36_feat'))
