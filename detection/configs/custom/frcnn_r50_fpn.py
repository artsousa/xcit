_base_ = [
    '_base_/models/faster_rcnn_r50_fpn.py',
    '_base_/datasets/coco_detection.py',
    '_base_/schedules/schedule_1x.py', 
    '_base_/default_runtime.py'
]

fp16 = None
im_size = (500, 500)

model = dict(
    type='FasterRCNN',
    backbone=dict(
        _delete_=True,
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    roi_head=dict(bbox_head=dict(num_classes=10))
)


optimizer_config = dict(grad_clip=dict(max_norm=25, norm_type=2))

fp16 = None
optimizer = dict(_delete_=True, type='AdamW', lr=0.002, betas=(0.9, 0.999), weight_decay=0.005)
lr_config = dict(step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)
optimizer_config = dict(grad_clip=dict(_delete_=True, max_norm=25, norm_type=2))

