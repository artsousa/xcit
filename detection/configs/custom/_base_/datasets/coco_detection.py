# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
dataset_type = 'CocoDataset'
data_root = '/home/arthurricardo98/UTI/'
img_norm_cfg = dict(mean=[0, 0, 0], std=[1, 1, 1], to_rgb=True)

classes=('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
img_prefix = data_root + '/digits/'
ann_test = data_root + '/annotations/digits/digits_test.json' 
ann_train = data_root + '/annotations/digits/digits_train.json' 

im_size = (500, 500)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False, poly2mask=False),
    dict(type='Resize', img_scale=im_size, keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=im_size,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
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

