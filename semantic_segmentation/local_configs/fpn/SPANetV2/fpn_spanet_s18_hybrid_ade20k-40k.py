_base_ = 'fpn_spanet_s18_pure_ade20k-40k.py'


# -- model settings
ckpt_file = "ckpt/spanetv2_s18_hybrid_k7_res-scale_Full-ExSPAM.pth"

model = dict(
    backbone=dict(
        type='spanetv2_s18_hybrid',
        drop_path_rate=0.15,
        head_dropout=0.0, 
        init_cfg=dict(
            type="Pretrained",
            checkpoint=ckpt_file
        ),
    ),
    neck=dict(in_channels=[64, 128, 320, 512]),
    decode_head=dict(
        num_classes=150,
    ),
)

