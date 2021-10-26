dataset_type = 'CocoDataset'
data_root = '/home/arthursvrr/MONITORS'

img_prefix = data_root + '/monitorsv2/'
ann_test = data_root + '/annotations/monitors_train/monitorsv2_test.json' 
ann_train = data_root + '/annotations/monitors_train/monitorsv2_train.json' 

pclasses = 1
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        #dict(type='TensorboardLoggerHook')
    ])
custom_hooks = [dict(type='NumClassCheckHook')]

classes=('screen',)
checkpoint_config = dict(interval=1)
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None # '/home/arthurricardo98/xcit/detection/outputs/epoch_20.pth'
workflow = [('train', 1)]
log_level = 'INFO'
evaluation = dict(interval=1, metric='bbox')
runner = dict(type='EpochBasedRunner', max_epochs=10)

fp16 = None
input_size = 512

optimizer_config = dict(grad_clip=dict(max_norm=25, norm_type=2))
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[1, 1, 1], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(
        type='Expand',
        mean=img_norm_cfg['mean'],
        to_rgb=img_norm_cfg['to_rgb'],
        ratio_range=(1, 4)),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
        min_crop_size=0.3),
    dict(type='Resize', img_scale=(input_size, input_size), keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(input_size, input_size),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file=ann_train,
        img_prefix=img_prefix,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=ann_test,
        img_prefix=img_prefix,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=ann_test,
        img_prefix=img_prefix,
        pipeline=test_pipeline))

optimizer = dict(type='AdamW', lr=0.002, betas=(0.9, 0.999), weight_decay=0.0005)
lr_config = dict(  
    policy='step', 
    warmup='linear', 
    warmup_iters=4000,
    warmup_ratio=0.001,
    step=[4, 7])


model = dict(
    type='SingleStageDetector',
    backbone=dict(
        type='ResNet',
        depth=34,
        num_stages=4,
        out_indices=(2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='SSDNeck',
        in_channels=(256, 512),
        out_channels=(256, 512, 512, 256, 256, 256, 256),
        level_strides=(2, 2, 2, 2, 1),
        level_paddings=(1, 1, 1, 1, 1),
        last_kernel_size=3,
        l2_norm_scale=20),
    bbox_head=dict(
        type='SSDHead',
        in_channels=(256, 512, 512, 256, 256, 256, 256),
        num_classes=pclasses,
        anchor_generator=dict(
            type='SSDAnchorGenerator',
            scale_major=False,
            input_size=input_size,
            basesize_ratio_range=(0.1, 0.9),
            strides=[8, 16, 32, 64, 128, 256, 512],
            ratios=[[2], [2, 3], [2, 3], [2, 3], [2, 3], [2], [2]]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2])),
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.,
            ignore_iof_thr=-1,
            gt_max_assign_all=False),
        smoothl1_beta=1.,
        allowed_border=-1,
        pos_weight=-1,
        neg_pos_ratio=3,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        nms=dict(type='nms', iou_threshold=0.45),
        min_bbox_size=0,
        score_thr=0.02,
        max_per_img=200))

    # model training and testing settings


