_base_ = 'upernet_spanet_s36_pure_ade20k-160k.py'

crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)

# -- model settings
ckpt_file = "ckpt/spanetv2_s36_hybrid_k7_res-scale_Full-ExSPAM.pth"

model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='spanetv2_s36_hybrid',
        drop_path_rate=0.3,
        head_dropout=0.4, 
        init_cfg=dict(
            type="Pretrained",
            checkpoint=ckpt_file
        ),
    ),
    decode_head=dict(
        in_channels=[64, 128, 320, 512],
        channels=320,
        num_classes=150,
    ),
    auxiliary_head=dict(in_channels=320, num_classes=150),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(341, 341)),
)
