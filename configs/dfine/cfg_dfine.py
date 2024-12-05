_base_ = [
    '../_base_/datasets/coco_detection.py', '../_base_/default_runtime.py'
]

model = dict(
    type='DFINE',
    # num_queries=100,
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=1),
    backbone=dict(
        type='HGNetv2',
        name='B0',
        use_lab=True,
        return_idx=[1, 2, 3],
        freeze_stem_only=True,
        freeze_at=-1,
        freeze_norm=False,
        pretrained=True
        ),
    encoder=dict(
        type='HybridEncoder',
        in_channels=[256, 512, 1024],
        feat_strides=[8, 16, 32],
        hidden_dim=256,
        nhead=8,
        dim_feedforward = 1024,
        dropout=0.0,
        enc_act='gelu',
        use_encoder_idx=[2],
        num_encoder_layers=1,
        pe_temperature=10000,
        expansion=1.0,
        depth_mult=0.34,
        act='silu',
        eval_spatial_size=None
    ),
    decoder=dict(
        type='DFINETransformer',
        num_classes=80,
        hidden_dim=256,
        num_queries=300,
        feat_channels=[256, 256, 256],
        feat_strides=[8, 16, 32],
        num_levels=3,
        num_points=[3, 6, 3],
        nhead=8,
        num_layers=3,
        dim_feedforward=1024,
        dropout=0.,
        activation="relu",
        num_denoising=100,
        label_noise_ratio=0.5,
        box_noise_scale=1.0,
        learn_query_content=False,
        eval_spatial_size=[640, 640],
        eval_idx=-1,
        eps=1e-2,
        aux_loss=True,
        cross_attn_method='default',
        query_select_method='default',
        reg_max=32,
        reg_scale=4.,
        layer_scale=1
    ),
    criterion=dict(
        type='DFINECriterion',
        weight_dict=dict(loss_vfl=1, 
                loss_bbox=5, 
                loss_giou=2, 
                loss_fgl=0.15, 
                loss_ddf=1.5),
        matcher=dict(
            type='HungarianMatcher',
                weight_dict=dict(cost_class=2, 
                    cost_bbox=5, 
                    cost_giou=2),
            alpha=0.25,
            gamma=2.0),
        losses=['vfl', 'boxes', 'local'],
        alpha=0.2,
        gamma=2.0,
        num_classes=80,
        reg_max=32,
        boxes_weight_format=None,
        share_matched_indices=False
    ),
    test_cfg=dict(max_per_img=100)
)

vis_backends = [
    dict(type='LocalVisBackend'),
]

visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer', save_dir='')


# train_pipeline, NOTE the img_scale and the Pad's size_divisor is different
# from the default setting in mmdet.
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Resize', scale=(640, 640)),
    dict(type='PackDetInputs')
]
train_dataloader = dict(dataset=dict(pipeline=train_pipeline), sampler=dict(type='InfiniteSampler', shuffle=False))

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={'backbone': dict(lr_mult=0.1, decay_mult=1.0)}))

# learning policy
# max_epochs = 155
# train_cfg = dict(
#     type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=150)
# val_cfg = dict(type='ValLoop')
# test_cfg = dict(type='TestLoop')

# learning policy
max_epochs = 150
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=1500, val_interval=500)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[100],
        gamma=0.1)
]

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (2 samples per GPU)
auto_scale_lr = dict(base_batch_size=16)