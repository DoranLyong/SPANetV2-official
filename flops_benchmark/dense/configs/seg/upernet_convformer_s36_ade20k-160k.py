# Constructed for the re-measurement (not in any repository):
# SPANetV2-S36-pure UperNet config with only the backbone replaced (decode/auxiliary heads unchanged).
_base_ = '../../../../semantic_segmentation/local_configs/upernet/SPANetV2/upernet_spanet_s36_pure_ade20k-160k.py'
model = dict(backbone=dict(_delete_=True, type='convformer_s36_feat'))
