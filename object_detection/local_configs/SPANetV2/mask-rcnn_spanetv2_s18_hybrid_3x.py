_base_ = 'mask-rcnn_spanetv2_s18_pure_3x.py'

# == dataset settings 
dataset_type = 'CocoDataset'
data_root = '/workspace/dataset/coco2017/'


# == model settings 
ckpt_file="ckpt/spanetv2_s18_hybrid_k7_res-scale_Full-ExSPAM.pth"


model = dict(
    backbone=dict(
        _delete_=True,
        type='spanetv2_s18_hybrid',
        drop_path_rate=0.15,
        head_dropout=0.0,           
        init_cfg=dict(
            type='Pretrained',
            checkpoint=ckpt_file        
        )
    ),
    neck=dict(in_channels=[64, 128, 320, 512],
    )
)