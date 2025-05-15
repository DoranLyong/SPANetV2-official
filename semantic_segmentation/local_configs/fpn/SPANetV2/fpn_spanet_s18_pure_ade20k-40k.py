_base_ = [
    '../../_base_/models/fpn_spanetv2.py',
    '../../_base_/datasets/ade20k.py',
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_40k.py',
]

# dataset settings
dataset_type = 'ADE20KDataset'
data_root = '/workspace/dataset/ade20k/ADEChallengeData2016'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(
        type='RandomResize',
        scale=(2048, 512),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 512), keep_ratio=True),
    dict(type='ResizeToMultiple', size_divisor=32),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='PackSegInputs')
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=50,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            data_prefix=dict(
                img_path='images/training',
                seg_map_path='annotations/training'),
            pipeline=train_pipeline)))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/validation',
            seg_map_path='annotations/validation'),
        pipeline=test_pipeline))
test_dataloader = val_dataloader
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

# -- model settings
ckpt_file = "ckpt/spanetv2_s18_pure_res-scale_Full-ExSPAM.pth"

model = dict(
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
    neck=dict(in_channels=[64, 128, 320, 512]),
    decode_head=dict(
        num_classes=150,
    ),
)


find_unused_parameters = True  # ref) https://mmengine.readthedocs.io/en/latest/common_usage/debug_tricks.html


# -- models are trained on 4 GPUs 
train_dataloader = dict(batch_size=8)  # total_batch = num_gpu * batch_size 
val_dataloader = dict(batch_size=1)
test_dataloader = val_dataloader



# -- optimizer settings 
# -- Customize the Effective Batch Size 32.
optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0002, weight_decay=0.0001),
    )

# -- Schedule
param_scheduler = [
    dict(
        type='PolyLR',
        power=0.9,
        begin=0,
        end=40000,
        eta_min=0.0,
        by_epoch=False,
    )
]
