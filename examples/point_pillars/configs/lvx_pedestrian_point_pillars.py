import itertools
import logging

from det3d.builder import build_box_coder
from det3d.utils.config_tool import get_downsample_factor

# norm_cfg = dict(type='SyncBN', eps=1e-3, momentum=0.01)
norm_cfg = None

tasks = [
    dict(num_class=1, class_names=["Pedestrian",],),
]

class_names = list(itertools.chain(*[t["class_names"] for t in tasks]))

# training and testing settings
target_assigner = dict(
    type="iou",
    anchor_generators=[
        dict(
            type="anchor_generator_range",
            sizes=[0.5, 0.8, 1.8],
            anchor_ranges=[-40, -40.0, 0.9, 40.0, 40.0, 0.9],
            strides=[0.4, 0.4, 0.0], # if generate only 1 z_center, z_stride will be ignored
            offsets=[-39.8, -39.8, 0.9], # origin_offset + strides / 2 TODO: offsets
            rotations=[0, 1.57],
            matched_threshold=0.5,
            unmatched_threshold=0.25,
            class_name="Pedestrian",
        ),
    ],
    sample_positive_fraction=-1,
    sample_size=512,
    region_similarity_calculator=dict(type="nearest_iou_similarity",),
    pos_area_threshold=-1,
    tasks=tasks,
)

box_coder = dict(
    type="ground_box3d_coder", n_dim=7, linear_dim=False, encode_angle_vector=False,
)

# model settings
model = dict(
    type="PointPillars",
    pretrained=None,
    reader=dict(
        num_input_features=3,
        type="PillarFeatureNet",
        num_filters=[64],
        with_distance=False,
        norm_cfg=norm_cfg,
    ),
    backbone=dict(type="PointPillarsScatter", ds_factor=1, norm_cfg=norm_cfg,),
    neck=dict(
        type="RPN_LVX",
        layer_nums=[3, 5, 5],
        ds_layer_strides=[1, 2, 2],
        ds_num_filters=[128, 128, 256],
        us_layer_strides=[1, 2, 4],
        us_num_filters=[256, 256, 256],
        num_input_features=64,
        norm_cfg=norm_cfg,
        logger=logging.getLogger("RPN"),
    ),
    bbox_head=dict(
        # type='RPNHead',
        type="MultiGroupHead",
        mode="3d",
        in_channels=sum([256,256,256]),  # this is linked to 'neck' us_num_filters
        norm_cfg=norm_cfg,
        tasks=tasks,
        weights=[1,],
        box_coder=build_box_coder(box_coder),
        encode_background_as_zeros=True,
        loss_norm=dict(
            type="NormByNumPositives", pos_cls_weight=1.0, neg_cls_weight=1.0,
        ),
        loss_cls=dict(type="SigmoidFocalLoss", alpha=0.25, gamma=2.0, loss_weight=0.1,),
        loss_iou=dict(type="DIoULoss", loss_weight=0.0,),
        use_sigmoid_score=True,
        loss_bbox=dict(
            type="WeightedSmoothL1Loss",
            sigma=3.0,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            codewise=True,
            loss_weight=2.0,
        ),
        encode_rad_error_by_sin=True,
        loss_aux=dict(
            type="WeightedSoftmaxClassificationLoss",
            name="direction_classifier",
            loss_weight=2.0,
        ),
        direction_offset=0.0,
    ),
)

assigner = dict(
    box_coder=box_coder,
    target_assigner=target_assigner,
    out_size_factor=get_downsample_factor(model),
)


train_cfg = dict(assigner=assigner)

test_cfg = dict(
    nms=dict(
        use_rotate_nms=True,
        use_multi_class_nms=False,
        nms_pre_max_size=1000,
        nms_post_max_size=30,
        nms_iou_threshold=0.1,
    ),
    score_threshold=0.2,
    post_center_limit_range=[-40, -40.0, -0.1, 40.0, 40.0, 2.5],
    max_per_img=100,
)

# dataset settings
dataset_type = "LvxDataset"
data_root = "/Extra/zhangmeng/3d_detection/BBOX_x2_track"

db_sampler = dict(
    type="GT-AUG",
    enable=True,
    db_info_path=data_root+"/dbinfos_train.pkl",
    sample_groups=[dict(Pedestrian=0,),],
    db_prep_steps=[
        dict(filter_by_min_num_points=dict(Pedestrian=1,)),
        dict(filter_by_difficulty=[-1],),
    ],
    global_random_rotation_range_per_object=[0, 0],
    rate=1.0,
)
train_preprocessor = dict(
    mode="train",
    shuffle_points=True,
    gt_loc_noise=[0.15, 0.15, 0.15],
    gt_rot_noise=[-0.15707963267, 0.15707963267],
    global_rot_noise=[-0.78539816, 0.78539816],
    global_scale_noise=[0.95, 1.05],
    global_rot_per_obj_range=[0, 0],
    global_trans_noise=[0.0, 0.0, 0.0],
    remove_points_after_sample=True,
    gt_drop_percentage=0.0,
    gt_drop_max_keep_points=15,
    remove_unknown_examples=False,
    remove_environment=False,
    db_sampler=db_sampler,
    class_names=class_names,
)

val_preprocessor = dict(
    mode="val",
    shuffle_points=False,
    remove_environment=False,
    remove_unknown_examples=False,
)

test_preprocessor = dict(
    mode="test",
    shuffle_points=False,
    remove_environment=False,
    remove_unknown_examples=False,
)

voxel_generator = dict(
    range=[-40, -40, -0.1, 40, 40, 2.0],
    voxel_size=[80./320, 80./320, 2.0],
    max_points_in_voxel=50,
    max_voxel_num=20000,
)

train_pipeline = [
    dict(type="LoadPointCloudFromFile", dataset=dataset_type),
    dict(type="LoadPointCloudAnnotations", with_bbox=True),
    dict(type="Preprocess", cfg=train_preprocessor),
    dict(type="Voxelization", cfg=voxel_generator),
    dict(type="AssignTarget", cfg=train_cfg["assigner"]),
    dict(type="Reformat"),
    # dict(type='PointCloudCollect', keys=['points', 'voxels', 'annotations', 'calib']),
]
val_pipeline = [
    dict(type="LoadPointCloudFromFile", dataset=dataset_type),
    dict(type="LoadPointCloudAnnotations", with_bbox=True),
    dict(type="Preprocess", cfg=val_preprocessor),
    dict(type="Voxelization", cfg=voxel_generator),
    dict(type="AssignTarget", cfg=train_cfg["assigner"]),
    dict(type="Reformat"),
]
test_pipeline = [
    dict(type="LoadPointCloudFromFile", dataset=dataset_type),
    # dict(type="LoadPointCloudAnnotations", with_bbox=True),
    dict(type="Preprocess", cfg=test_preprocessor),
    dict(type="Voxelization", cfg=voxel_generator),
    dict(type="AssignTarget", cfg=train_cfg["assigner"]),
    dict(type="Reformat"),
]

train_anno = data_root+"/lvx_infos_train.pkl"
val_anno = data_root+"/lvx_infos_val.pkl"
test_anno = None
start_idx = [[0,115],[117,332],[333,560]] # 训练集为三段

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=data_root + "/lvx_infos_train.pkl",
        ann_file=train_anno,
        class_names=class_names,
        start_idx = start_idx,
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=data_root + "/lvx_infos_val.pkl",
        ann_file=val_anno,
        class_names=class_names,
        pipeline=val_pipeline,
    ),
    test=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=data_root + "/lvx_infos_test.pkl",
        ann_file=test_anno,
        class_names=class_names,
        pipeline=test_pipeline,
    ),
)

# optimizer = dict(type='SGD', lr=0.0002, momentum=0.9, weight_decay=0.0001)
# optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# # learning policy
# lr_config = dict(
#     type='multinomial',
#     policy='step',
#     warmup='linear',
#     warmup_iters=500,
#     warmup_ratio=1.0 / 3,
#     step=[15, 30, 45, 40, 75, 90, 105, 120, 135, 150])

# optimizer
optimizer = dict(
    type="adam", amsgrad=0.0, wd=0.01, fixed_wd=True, moving_average=False,
)

"""training hooks """
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy in training hooks
lr_config = dict(
    type="one_cycle", lr_max=3e-3, moms=[0.95, 0.85], div_factor=10.0, pct_start=0.4,
)

checkpoint_config = dict(interval=5)
# yapf:disable
log_config = dict(
    interval=20,
    hooks=[
        dict(type="TextLoggerHook"),
        # dict(type='TensorboardLoggerHook')
    ],
)
# yapf:enable
# runtime settings
total_epochs = 150
device_ids = range(8)
dist_params = dict(backend="nccl", init_method="env://")
log_level = "INFO"
work_dir = "/Extra/zhangmeng/Outputs/det3d_Outputs/Point_Pillars"
load_from = None
resume_from = None
workflow = [("train", 5),("val",1)]
