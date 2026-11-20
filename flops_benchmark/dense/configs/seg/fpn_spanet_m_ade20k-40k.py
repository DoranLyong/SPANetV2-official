# Constructed for the re-measurement: SPANet-official has no segmentation code.
# SPANetV2-S36-pure Semantic FPN config with only the backbone replaced by SPANet-M (layers [6, 6, 18, 6]:
# object_detection/local_configs/spanet/S36 and image_classification/spanet.py: spanet_medium); see fpn_spanet_s.
_base_ = '../../../../semantic_segmentation/local_configs/fpn/SPANetV2/fpn_spanet_s36_pure_ade20k-40k.py'
model = dict(backbone=dict(
    _delete_=True, type='SPANet',
    layers=[6, 6, 18, 6], embed_dims=[64, 128, 320, 512],
    patch_dims=[512 // 2**2, 512 // 2**3, 512 // 2**4, 512 // 2**5],
    radius=[2**1, 2**1, 2**0, 2**0], mlp_ratios=[4, 4, 4, 4], downsamples=[True, True, True, True]))
