# Constructed for the re-measurement: SPANet-official has no segmentation code.
# SPANetV2-S18-pure Semantic FPN config with only the backbone replaced by SPANet-S, i.e. the SPANet
# detection backbone (SPANet-official/object_detection/mmdet/models/backbones/spanet.py, class SPANet) with
# layers/embed_dims/radius/mlp_ratios/downsamples of SPANet-S (object_detection/local_configs/_base_/models/
# mask_rcnn_spanet_fpn.py and image_classification/spanet.py: spanet_small) and patch_dims for a 512x512 input
# (the img_size // 2**(i+2) rule of spanet.py; the backbone file's own __main__ uses img_size = 512).
_base_ = '../../../../semantic_segmentation/local_configs/fpn/SPANetV2/fpn_spanet_s18_pure_ade20k-40k.py'
model = dict(backbone=dict(
    _delete_=True, type='SPANet',
    layers=[4, 4, 12, 4], embed_dims=[64, 128, 320, 512],
    patch_dims=[512 // 2**2, 512 // 2**3, 512 // 2**4, 512 // 2**5],
    radius=[2**1, 2**1, 2**0, 2**0], mlp_ratios=[4, 4, 4, 4], downsamples=[True, True, True, True]))
