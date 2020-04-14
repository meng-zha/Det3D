#!/bin/bash
TASK_DESC=$1
DATE_WITH_TIME=`date "+%Y%m%d-%H%M%S"`
OUT_DIR='/Extra/zhangmeng/3d_detection/Outputs/Det3D_Outputs'

NUSC_CBGS_WORK_DIR=$OUT_DIR/NUSC_CBGS_$TASK_DESC\_$DATE_WITH_TIME
LYFT_CBGS_WORK_DIR=$OUT_DIR/LYFT_CBGS_$TASK_DESC\_$DATE_WITH_TIME
SECOND_WORK_DIR=$OUT_DIR/SECOND_$TASK_DESC\_$DATE_WITH_TIME
PP_WORK_DIR=$OUT_DIR/PointPillars_$TASK_DESC\_$DATE_WITH_TIME

if [ ! $TASK_DESC ] 
then
    echo "TASK_DESC must be specified."
    echo "Usage: train.sh task_description"
    exit $E_ASSERT_FAILED
fi

# Voxelnet
# python -m torch.distributed.launch --nproc_per_node=8 ./tools/train.py examples/second/configs/kitti_car_vfev3_spmiddlefhd_rpn1_mghead_syncbn.py --work_dir=$SECOND_WORK_DIR
# python -m torch.distributed.launch --nproc_per_node=8 ./tools/train.py examples/cbgs/configs/nusc_all_vfev3_spmiddleresnetfhd_rpn2_mghead_syncbn.py --work_dir=$NUSC_CBGS_WORK_DIR
# python -m torch.distributed.launch --nproc_per_node=8 ./tools/train.py examples/second/configs/lyft_all_vfev3_spmiddleresnetfhd_rpn2_mghead_syncbn.py --work_dir=$LYFT_CBGS_WORK_DIR

# PointPillars
# python -m torch.distributed.launch --nproc_per_node=1 ./tools/train.py ./examples/point_pillars/configs/kitti_pedestrian_point_pillars_mghead_syncbn.py --work_dir=$PP_WORK_DIR --local_rank=2
# python -m torch.distributed.launch --nproc_per_node=1 ./tools/train.py ./examples/point_pillars/configs/kitti_point_pillars_mghead_syncbn.py  --work_dir=$PP_WORK_DIR 
# python ./tools/train.py --local_rank=0 --work_dir=$PP_WORK_DIR ./examples/point_pillars/configs/kitti_pedestrian_point_pillars_mghead_syncbn.py --resume_from=Outputs/Det3D_Outputs/PointPillars_point_pillars_pedestrian_20200211-140512/epoch_99.pth
# CUDA_VISIBLE_DEVICES=2 python ./tools/train.py --work_dir=$PP_WORK_DIR --local_rank=0 ./examples/point_pillars/configs/kitti_point_pillars_mghead_syncbn.py 
CUDA_VISIBLE_DEVICES=2 python -W ignore ./tools/train.py --work_dir=$PP_WORK_DIR --local_rank=0 ./examples/point_pillars/configs/lvx_pedestrian_point_pillars.py 