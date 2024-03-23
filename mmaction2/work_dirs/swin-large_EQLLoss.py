# Copyright (c) UD_lab. All rights reserved.
_base_ = [
    "../configs/recognition/swin/swin-large-p244-w877_in22k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb.py"
]

import pandas as pd

model = dict(
    cls_head=dict(
        num_classes=140,
        loss_cls=dict(
            type="EQLLossWithLogits",
            class_counts=list(
                pd.read_excel(
                    "/root/autodl-tmp/AnimalKingdom/df_action.xlsx"
                ).sort_values("index")["count"]
            ),
        ),
    )
)

# 数据集配置
dataset_type = "VideoDataset"
dataset_path = "../../data/AnimalKingdom/"
data_root = dataset_path + "dataset/videos"
data_root_val = dataset_path + "dataset/videos"
ann_file_train = dataset_path + "videos_train.txt"
ann_file_val = dataset_path + "videos_val.txt"
ann_file_test = dataset_path + "videos_val.txt"

train_dataloader = dict(
    batch_size=2,
    dataset=dict(
        ann_file=ann_file_train,
        data_prefix=dict(video=data_root),
        multi_class=True,
        num_classes=140,
    ),
)

val_dataloader = dict(
    batch_size=2,
    dataset=dict(
        ann_file=ann_file_val,
        data_prefix=dict(video=data_root),
        multi_class=True,
        num_classes=140,
        test_mode=True,
    ),
)

test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file=ann_file_test,
        data_prefix=dict(video=data_root),
        multi_class=True,
        num_classes=140,
        test_mode=True,
    ),
)

val_evaluator = [
    dict(
        type="AccMetric",
        metric_list=("mean_average_precision", "mmit_mean_average_precision"),
    ),
    dict(
        type="AnimalKingdomMetric",
    ),
]

test_evaluator = [
    dict(
        type="AccMetric",
        metric_list=("mean_average_precision", "mmit_mean_average_precision"),
    ),
    dict(
        type="AnimalKingdomMetric",
    ),
]


# 训练配置
train_cfg = dict(max_epochs=10, type="EpochBasedTrainLoop", val_begin=1, val_interval=1)

optim_wrapper = dict(
    type="AmpOptimWrapper",
    optimizer=dict(type="AdamW", lr=1e-3, betas=(0.9, 0.999), weight_decay=0.05),
    constructor="SwinOptimWrapperConstructor",
    paramwise_cfg=dict(
        absolute_pos_embed=dict(decay_mult=0.0),
        relative_position_bias_table=dict(decay_mult=0.0),
        norm=dict(decay_mult=0.0),
        backbone=dict(lr_mult=0.1),
    ),
)

param_scheduler = [
    dict(
        type="LinearLR",
        start_factor=0.1,
        by_epoch=True,
        begin=0,
        end=1,
        convert_to_iter_based=True,
    ),
    dict(type="CosineAnnealingLR", T_max=20, eta_min=0, by_epoch=True, begin=1, end=10),
]

auto_scale_lr = dict(base_batch_size=12, enable=None)
load_from = "/root/autodl-tmp/swin-large_checkpoints/5263_6577_5861_4773.pth"
resume = False

# 可视化配置
visualizer = dict(
    type="Visualizer",
    vis_backends=[dict(type="TensorboardVisBackend", save_dir="/root/tf-logs")],
)
