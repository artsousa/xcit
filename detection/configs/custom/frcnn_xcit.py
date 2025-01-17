# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Hyperparameters modifed from
https://github.com/SwinTransformer/Swin-Transformer-Object-Detection
"""

pclasses = 11
dataset_type = 'CocoDataset'
data_root = '/home/arthursvrr/UTI'
_base_ = [
    '../_base_/models/mask_rcnn_xcit_p8.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

classes=('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'screen')

model = dict(
    pretrained=None,
    type='FasterRCNN',
    backbone=dict(
        type='XCiT',
        num_classes=pclasses,
        patch_size=8,
        embed_dim=192,
        depth=12,
        num_heads=4,
        mlp_ratio=4,
        qkv_bias=True,
        eta=1.0,
        drop_path_rate=0.0,
        out_indices=[3, 5, 7, 11]
    ),
    neck=dict(
        type='FPN',
        in_channels=[192, 192, 192, 192],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
      type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)
    ),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        mask_head=None,#dict(type='FCNMaskHead', num_classes=pclasses),
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
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100)
        # soft-nms is also supported for rcnn testing
        # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
    )
)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# augmentation strategy originates from DETR / Sparse RCNN
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False, poly2mask=False),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(type='Normalize',
         mean=[123.675, 116.28, 103.53],
         std=[58.395, 57.12, 57.375], to_rgb=True
        ),
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
            dict(type='RandomFlip', flip_ratio=0.0),
            dict(type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375], 
                to_rgb=True,
            ),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
        samples_per_gpu=2,
        workers_per_gpu=1,
        train=dict(
            type=dataset_type,
            classes=classes,
            ann_file=data_root + '/annotations/trainW.json',
            img_prefix=data_root + '/train2017/',
            pipeline=train_pipeline
            ),
        val=dict(
            type=dataset_type,
            classes=classes,
            ann_file=data_root + '/annotations/testW.json',
            img_prefix=data_root + '/train2017/',
            pipeline=test_pipeline),
        test=dict(
            type=dataset_type,
            classes=classes,
            ann_file=data_root + '/annotations/testW.json',
            img_prefix=data_root + '/train2017/',
            pipeline=test_pipeline) 
        )

optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))
lr_config = dict(  
    policy='step', 
    warmup='linear', 
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])

runner = dict(type='EpochBasedRunner', max_epochs=18)
#runner = dict(type='IterBasedRunner', max_iters=100)

checkpoint_config = dict(interval=1)
log_config = dict( 
    interval=100,
    hooks=[
        dict(type='TextLoggerHook')
    ])
log_level = 'INFO'
load_from = None
resume_from = None#'/home/arthursvrr/xcit/detection/outputs/epoch_14.pth' 
workflow = [('train', 1)]
evaluation = dict(interval=1, metric=['bbox'])

# do not use mmdet version fp16
fp16 = None
optimizer_config = dict(
    grad_clip=None,
)
