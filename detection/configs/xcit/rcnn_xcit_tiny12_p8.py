_base_ = [
    '_base_/models/mask_rcnn_xcit_p16.py',
    '_base_/datasets/coco_detection.py',
    '_base_/schedules/schedule_1x.py', 
    '_base_/default_runtime.py'
]

data = dict(samples_per_gpu=4, 
            workers_per_gpu=1)

# small config
model = dict(
    backbone=dict(
        type='XCiT',
        patch_size=8,
        num_classes=10,
        embed_dim=384,
        depth=12,
        num_heads=8,
        mlp_ratio=4,
        qkv_bias=True,
        eta=1.0,
        drop_path_rate=0.05,
        out_indices=[3, 5, 7, 11],
    ),
    neck=dict(in_channels=[384, 384, 384, 384]),
    roi_head=dict(bbox_head=dict(num_classes=10))
)  

optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))

lr_config = dict(step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)
fp16 = None
optimizer_config = dict(grad_clip=None)
