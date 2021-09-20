# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Hyperparameters modifed from
https://github.com/SwinTransformer/Swin-Transformer-Object-Detection
"""

pclasses = 11
dataset_type = 'CocoDataset'
data_root = '/home/koda/code-linux/data/uti-all/'
_base_ = [
    '../_base_/models/mask_rcnn_xcit_p8.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

classes=('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'screen')
log_level='INFO'

model = dict(
    pretrained=None,
    backbone=dict(
        type='XCiT',
        num_classes=pclasses,
        patch_size=8,
        embed_dim=384,
        depth=12,
        num_heads=8,
        mlp_ratio=4,
        qkv_bias=True,
        eta=1.0,
        drop_path_rate=0.05,
        out_indices=[3, 5, 7, 11]
    ),
    neck=dict(in_channels=[384, 384, 384, 384]),
    roi_head=dict(
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=pclasses,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)
            ),
        mask_head=dict(
            type='FCNMaskHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=pclasses,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))
    ), 
    
)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# augmentation strategy originates from DETR / Sparse RCNN
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        flip=False,
        img_scale=(1333, 800),
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
        samples_per_gpu=1,
        workers_per_gpu=1,
        train=dict(
            type=dataset_type,
            classes=classes,
            ann_file=data_root + 'annotations/train_splited.json',
            img_prefix=data_root + 'train2017/',
            pipeline=train_pipeline
            ),
        val=dict(
            type=dataset_type,
            classes=classes,
            ann_file=data_root + 'annotations/test_splited.json',
            img_prefix=data_root + 'train2017/',
            pipeline=test_pipeline),
        #test=dict(
        #    type=dataset_type,
        #    classes=classes,
        #    ann_file=data_root + 'annotations/test_splited.json',
        #    img_prefix=data_root + 'train2017/',
        #    pipeline=test_pipeline) 
        )

optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))
#lr_config = dict(step=[27, 33])
runner = dict(type='EpochBasedRunner', max_epochs=2)
#runner = dict(type='IterBasedRunner', max_iters=2)

# do not use mmdet version fp16
#fp16 = None
#optimizer_config = dict(
#    type="DistOptimizerHook",
#    update_interval=1,
#    grad_clip=None,
#    coalesce=True,
#    bucket_size_mb=-1,
#    use_fp16=fp16,
#)
