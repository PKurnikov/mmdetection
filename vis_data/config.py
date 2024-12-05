auto_scale_lr = dict(base_batch_size=16)
backend_args = None
data_root = 'data/coco/'
dataset_type = 'CocoDataset'
default_hooks = dict(
    checkpoint=dict(interval=500, type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='DetVisualizationHook'))
default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
max_epochs = 150
model = dict(
    backbone=dict(
        freeze_at=-1,
        freeze_norm=False,
        freeze_stem_only=True,
        name='B0',
        pretrained=True,
        return_idx=[
            1,
            2,
            3,
        ],
        type='HGNetv2',
        use_lab=True),
    criterion=dict(
        alpha=0.2,
        boxes_weight_format=None,
        gamma=2.0,
        losses=[
            'vfl',
            'boxes',
            'local',
        ],
        matcher=dict(
            alpha=0.25,
            gamma=2.0,
            type='HungarianMatcher',
            weight_dict=dict(cost_bbox=5, cost_class=2, cost_giou=2)),
        num_classes=80,
        reg_max=32,
        share_matched_indices=False,
        type='DFINECriterion',
        weight_dict=dict(
            loss_bbox=5, loss_ddf=1.5, loss_fgl=0.15, loss_giou=2,
            loss_vfl=1)),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_size_divisor=1,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='DetDataPreprocessor'),
    decoder=dict(
        activation='relu',
        aux_loss=True,
        box_noise_scale=1.0,
        cross_attn_method='default',
        dim_feedforward=1024,
        dropout=0.0,
        eps=0.01,
        eval_idx=-1,
        eval_spatial_size=[
            640,
            640,
        ],
        feat_channels=[
            256,
            256,
            256,
        ],
        feat_strides=[
            8,
            16,
            32,
        ],
        hidden_dim=256,
        label_noise_ratio=0.5,
        layer_scale=1,
        learn_query_content=False,
        nhead=8,
        num_classes=80,
        num_denoising=100,
        num_layers=3,
        num_levels=3,
        num_points=[
            3,
            6,
            3,
        ],
        num_queries=300,
        query_select_method='default',
        reg_max=32,
        reg_scale=4.0,
        type='DFINETransformer'),
    encoder=dict(
        act='silu',
        depth_mult=0.34,
        dim_feedforward=1024,
        dropout=0.0,
        enc_act='gelu',
        eval_spatial_size=None,
        expansion=1.0,
        feat_strides=[
            8,
            16,
            32,
        ],
        hidden_dim=256,
        in_channels=[
            256,
            512,
            1024,
        ],
        nhead=8,
        num_encoder_layers=1,
        pe_temperature=10000,
        type='HybridEncoder',
        use_encoder_idx=[
            2,
        ]),
    test_cfg=dict(max_per_img=100),
    type='DFINE')
optim_wrapper = dict(
    clip_grad=dict(max_norm=0.1, norm_type=2),
    optimizer=dict(lr=0.0001, type='AdamW', weight_decay=0.0001),
    paramwise_cfg=dict(
        custom_keys=dict(backbone=dict(decay_mult=1.0, lr_mult=0.1))),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0,
        by_epoch=True,
        end=150,
        gamma=0.1,
        milestones=[
            100,
        ],
        type='MultiStepLR'),
]
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=8,
    dataset=dict(
        ann_file='annotations/instances_train2017.json',
        backend_args=None,
        data_prefix=dict(img='train2017/'),
        data_root='data/coco/',
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=False, scale=(
                640,
                640,
            ), type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file='data/coco/annotations/instances_train2017.json',
    backend_args=None,
    format_only=False,
    metric='bbox',
    type='CocoMetric')
test_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(keep_ratio=False, scale=(
        640,
        640,
    ), type='Resize'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackDetInputs'),
]
train_cfg = dict(max_iters=1500, type='IterBasedTrainLoop', val_interval=500)
train_dataloader = dict(
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    batch_size=8,
    dataset=dict(
        ann_file='annotations/instances_train2017.json',
        backend_args=None,
        data_prefix=dict(img='train2017/'),
        data_root='data/coco/',
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(prob=0.5, type='RandomFlip'),
            dict(scale=(
                640,
                640,
            ), type='Resize'),
            dict(type='PackDetInputs'),
        ],
        type='CocoDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='InfiniteSampler'))
train_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(prob=0.5, type='RandomFlip'),
    dict(scale=(
        640,
        640,
    ), type='Resize'),
    dict(type='PackDetInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=8,
    dataset=dict(
        ann_file='annotations/instances_train2017.json',
        backend_args=None,
        data_prefix=dict(img='train2017/'),
        data_root='data/coco/',
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=False, scale=(
                640,
                640,
            ), type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file='data/coco/annotations/instances_train2017.json',
    backend_args=None,
    format_only=False,
    metric='bbox',
    type='CocoMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    save_dir='',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './work_dirs/cfg_dfine'
