_base_ = [
    '_base_/models/mask_rcnn_xcit_p16.py',
    '_base_/datasets/coco_detection.py',
    '_base_/schedules/schedule_1x.py', 
    '_base_/default_runtime.py'
]

data = dict(samples_per_gpu=2, 
            workers_per_gpu=1)

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(type='AutoAugment',
        policies=[
             #[
             #    dict(type='Resize',
             #         img_scale=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
             #                    (608, 1333), (640, 1333), (672, 1333), (704, 1333),
             #                    (736, 1333), (768, 1333), (800, 1333)],
             #         multiscale_mode='value',
             #         keep_ratio=True)
             #],
             [
                 dict(type='Resize',
                      img_scale=[(400, 600), (500, 600), (600, 600)],
                      multiscale_mode='value',
                      keep_ratio=True),
                 dict(type='RandomCrop',
                      crop_type='absolute_range',
                      crop_size=(384, 600),
                      allow_negative_crop=True),
                 dict(type='Resize',
                      img_scale=[(480, 600), (512, 600), (544, 600)],
                      multiscale_mode='value',
                      override=True,
                      keep_ratio=True)
             ]
         ]), 
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
data = dict(train=dict(pipeline=train_pipeline))

# small config
model = dict(
    backbone=dict(
        type='XCiT',
        patch_size=16,
        num_classes=10,
        embed_dim=192,
        depth=12,
        num_heads=4,
        mlp_ratio=4,
        qkv_bias=True,
        eta=1.0,
        drop_path_rate=0.05,
        out_indices=[3, 5, 7, 11],
    ),
    neck=dict(in_channels=[192, 192, 192, 192]),
    roi_head=dict(bbox_head=dict(num_classes=10))
)  

optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))

lr_config = dict(step=[16, 21])
runner = dict(type='EpochBasedRunner', max_epochs=22)
fp16 = None
optimizer_config = dict(grad_clip=None)
