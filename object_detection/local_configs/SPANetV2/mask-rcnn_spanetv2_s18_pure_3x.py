_base_ = [
    '../_base_/models/mask-rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]


# == dataset settings 
dataset_type = 'CocoDataset'
data_root = '/workspace/dataset/coco2017/'


# == model settings 
ckpt_file="ckpt/spanetv2_s18_pure_res-scale_Full-ExSPAM.pth"


model = dict(
    backbone=dict(
        _delete_=True,
        type='spanetv2_s18_pure',
        drop_path_rate=0.2,
        head_dropout=0.0,         
        init_cfg=dict(
            type='Pretrained',
            checkpoint=ckpt_file        
        )
    ),
    neck=dict(in_channels=[64, 128, 320, 512],
    )
)


find_unused_parameters = True  # ref) https://mmengine.readthedocs.io/en/latest/common_usage/debug_tricks.html


# == augmentation strategy originates from DETR / Sparse RCNN
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[[
            dict(
                type='RandomChoiceResize',
                scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                        (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                        (736, 1333), (768, 1333), (800, 1333)],
                keep_ratio=True)
        ],
                    [
                        dict(
                            type='RandomChoiceResize',
                            scales=[(400, 1333), (500, 1333), (600, 1333)],
                            keep_ratio=True),
                        dict(
                            type='RandomCrop',
                            crop_type='absolute_range',
                            crop_size=(384, 600),
                            allow_negative_crop=True),
                        dict(
                            type='RandomChoiceResize',
                            scales=[(480, 1333), (512, 1333), (544, 1333),
                                    (576, 1333), (608, 1333), (640, 1333),
                                    (672, 1333), (704, 1333), (736, 1333),
                                    (768, 1333), (800, 1333)],
                            keep_ratio=True)
                    ]]),
    dict(type='PackDetInputs')
]

# -- We use 4 GPUs.
train_dataloader = dict(
                    batch_size=2, # total_batch = num_gpu * batch_size 
                    dataset=dict(pipeline=train_pipeline), 
                    persistent_workers=True)

max_epochs = 36   # 3 x 12(epoch)
train_cfg = dict(max_epochs=max_epochs)

# == learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0,
        end=1000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[27, 33],
        gamma=0.1)
]

# == Enable automatic-mixed-precision training with AmpOptimWrapper.
# -- Customize the 'accumulative_counts' to be sure the Effective Batch Size 16.
optim_wrapper = dict(
    type='AmpOptimWrapper',
    constructor='SPANetV2LearningRateDecayOptimizerConstructor',
    paramwise_cfg={
        'decay_rate': 0.95,
        'decay_type': 'layer_wise',
        'num_layers': 6,  # num_stage
    },
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=1e-4, # like ConvNext
        #lr=3e-4, # like RDNet
        betas=(0.9, 0.999),
        weight_decay=0.05),
    accumulative_counts=2 # Effective Batch Size = num_gpu * batch_size * accumulative_counts
    )