# Constructed for the re-measurement: object_detection/local_configs/SPANetV2 has only the S18 Cascade configs.
# S18-pure Cascade Mask R-CNN 3x config with only the backbone name changed to S36
# (S36 has the same stage channels [64, 128, 320, 512] as S18; neck, heads, test_cfg unchanged).
_base_ = '../../../../object_detection/local_configs/SPANetV2/cascade-mask-rcnn_spanetv2_s18_pure_fpn-3x_coco.py'
model = dict(backbone=dict(type='spanetv2_s36_pure'), neck=dict(in_channels=[64, 128, 320, 512]))
