_base_ = [
    '../../_base_/models/upernet_spanetv2.py',
    '../../_base_/datasets/ade20k.py',
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_160k.py',
]

crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)


# -- model settings
ckpt_file = "./ckpt/spanetv2_s18_pure_res-scale_Full-ExSPAM.pth"

model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='spanetv2_s18_pure',
        drop_path_rate=0.2,
        head_dropout=0.0, 
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


find_unused_parameters = True  # ref) https://mmengine.readthedocs.io/en/latest/common_usage/debug_tricks.html


# -- optimizer settings 
optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW', 
        lr=1e-4, # like ConvNext
        betas=(0.9, 0.999), weight_decay=0.05),
    paramwise_cfg={
        'decay_rate': 0.9,
        'decay_type': 'stage_wise',
        'num_layers': 7
    },
    constructor='SPANetV2LearningRateDecayOptimizerConstructor',
    loss_scale='dynamic')

# -- For total 16-batch, models are trained on 4 GPUs with 4 images per GPU
train_dataloader = dict(batch_size=4)  # total_batch = num_gpu * batch_size 
val_dataloader = dict(batch_size=1)
test_dataloader = val_dataloader


# -- Schedule
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        power=1.0,
        begin=1500,
        end=160000,
        eta_min=0.0,
        by_epoch=False,
    )
]
