import numpy as np
import pickle
import os

from copy import deepcopy

from det3d.core import box_np_ops
from det3d.datasets.custom import PointCloudDataset
from det3d.datasets.registry import DATASETS

from .lvx_common import *
from .eval import get_official_eval_result
from .lvx_vis import lvx_vis
from .lvx_track import lvx_track


@DATASETS.register_module
class LvxDataset(PointCloudDataset):

    NumPointFeatures = 3

    def __init__(
        self,
        root_path,
        info_path,
        cfg=None,
        pipeline=None,
        class_names=None,
        test_mode=False,
        start_idx = [],
        **kwargs
    ):
        self._start_idx = start_idx 
        super(LvxDataset, self).__init__(
            root_path, info_path, pipeline, test_mode=test_mode
        )
        assert self._info_path is not None
        if not hasattr(self, "_lvx_infos"):
            self._lvx_infos = self.load_infos(self._info_path)
        self._num_point_features = __class__.NumPointFeatures
        # print("remain number of infos:", len(self._lvx_infos))
        self._class_names = class_names
        if self._start_idx == []:
            self._start_idx=[[0,len(self._lvx_infos)-1]]
        self._video_times = self.load_videos(self._start_idx)
        self.raw_length = len(self._lvx_infos)
    
    def load_infos(self,info_path):
        
        with open(self._info_path,"rb") as f:
            return pickle.load(f)

    def load_videos(self,start_idx):
        if not hasattr(self, "_lvx_infos"):
            self._lvx_infos = self.load_infos(self._info_path)
        if self._start_idx == []:
            self._start_idx=[[0,len(self._lvx_infos)-1]]
        video_times = [0]
        length = 0
        for video_time in self._start_idx:
            length += video_time[1]-video_time[0] -2
            video_times.append(length)
        return video_times

    def __len__(self):

        # if not hasattr(self, "_lvx_infos"):
        #     self._lvx_infos = self.load_infos(self._info_path)
        if not hasattr(self, "_video_times"):
            self._video_times = self.load_videos(self._start_idx)

        return self._video_times[-1]

    @property
    def num_point_features(self):
        return self._num_point_features

    @property
    def ground_truth_annotations(self):
        if "annos" not in self._lvx_infos[0]:
            return None

        lvx_infos = []
        for clips in self._start_idx:
            lvx_infos.extend(self._lvx_infos[clips[0]+2:clips[1]])

        gt_annos = [info["annos"] for info in lvx_infos]

        return gt_annos

    def convert_detection_to_lvx_annos(self, detection):
        class_names = self._class_names
        lvx_infos = []
        for clips in self._start_idx:
            lvx_infos.extend(self._lvx_infos[clips[0]+2:clips[1]])
        gt_image_idxes = [str(info["token"]) for info in lvx_infos]
        annos = []
        for det_idx in gt_image_idxes:
            det = detection[det_idx]
            info = self._lvx_infos[gt_image_idxes.index(det_idx)]
            # info = self._lvx_infos[det_idx]
            final_box_preds = det["box3d_lidar"].detach().cpu().numpy()
            final_box_preds_1 = det["box3d_lidar_1"].detach().cpu().numpy()
            final_box_preds_2 = det["box3d_lidar_2"].detach().cpu().numpy()
            label_preds = det["label_preds"].detach().cpu().numpy()
            scores = det["scores"].detach().cpu().numpy()

            anno = get_start_result_anno()
            num_example = 0

            if final_box_preds.shape[0] != 0:
                final_box_preds[:, -1] = box_np_ops.limit_period(
                    final_box_preds[:, -1], offset=0.5, period=np.pi * 2,
                )
                final_box_preds_1[:, -1] = box_np_ops.limit_period(
                    final_box_preds_1[:, -1], offset=0.5, period=np.pi * 2,
                )
                final_box_preds_2[:, -1] = box_np_ops.limit_period(
                    final_box_preds_2[:, -1], offset=0.5, period=np.pi * 2,
                )
                bbox = np.asarray([0,0,500,500])
                for j in range(final_box_preds.shape[0]):
                    anno["bbox"].append(bbox)
                    anno["alpha"].append(-10)
                    anno["dimensions"].append(final_box_preds[j, 3:6])
                    anno["location"].append(final_box_preds[j, :3])
                    anno["rotation_y"].append(final_box_preds[j, 6])
                    anno["dimensions_1"].append(final_box_preds_1[j, 3:6])
                    anno["location_1"].append(final_box_preds_1[j, :3])
                    anno["rotation_y_1"].append(final_box_preds_1[j, 6])
                    anno["dimensions_2"].append(final_box_preds_2[j, 3:6])
                    anno["location_2"].append(final_box_preds_2[j, :3])
                    anno["rotation_y_2"].append(final_box_preds_2[j, 6])
                    anno["name"].append(class_names[int(label_preds[j])])
                    anno["truncated"].append(0.0)
                    anno["occluded"].append(0)
                    anno["score"].append(scores[j])

                    num_example+=1
            
            if num_example != 0:
                anno = {n: np.stack(v) for n, v in anno.items()}
                annos.append(anno)
            else:
                annos.append(empty_result_anno())
            num_example = annos[-1]["name"].shape[0]
            annos[-1]["metadata"] = det["metadata"]
        return annos

    def evaluation(self, detections, output_dir=None, vis = False, track = False):
        """
        detection
        When you want to eval your own dataset, you MUST set correct
        the z axis and box z center.
        """

        import copy

        gt_source = self.ground_truth_annotations
        gt_annos = copy.deepcopy(gt_source)
        dt_source = self.convert_detection_to_lvx_annos(detections)
        dt_annos = copy.deepcopy(dt_source)

        if track:
            # 存入跟踪结果
            dt_annos = lvx_track(dt_annos)

        if vis:
            lvx_vis(gt_annos,dt_annos,output_dir)

        # firstly convert standard detection to lvx-format dt annos
        z_axis = 2  # KITTI camera format use y as regular "z" axis.
        z_center = 0.5  # KITTI camera box's center is [0.5, 1, 0.5]
        # for regular raw lidar data, z_axis = 2, z_center = 0.5.

        if gt_annos is not None:
            result_official_dict = get_official_eval_result(
                gt_annos, dt_annos, self._class_names, z_axis=z_axis, z_center=z_center,difficultys=[0]
            )

            def change_time(gt_annos,dt_annos,time):
                ''' 评价T-1帧和T+1帧的检测框'''
                for i in range(len(gt_annos)):
                    gt_annos[i]['dimensions'] = gt_annos[i][f'dimensions_{time}']
                    gt_annos[i]['location'] = gt_annos[i][f'location_{time}']
                    gt_annos[i]['rotation_y'] = gt_annos[i][f'rotation_y_{time}']
                    dt_annos[i]['dimensions'] = dt_annos[i][f'dimensions_{time}']
                    dt_annos[i]['location'] = dt_annos[i][f'location_{time}']
                    dt_annos[i]['rotation_y'] = dt_annos[i][f'rotation_y_{time}']
                return gt_annos,dt_annos

            gt_annos_1,dt_annos_1 = change_time(gt_annos,dt_annos,1)
            result_official_dict_1 = get_official_eval_result(
                gt_annos_1, dt_annos_1, self._class_names, z_axis=z_axis, z_center=z_center,difficultys=[0]
            )

            gt_annos_2,dt_annos_2 = change_time(gt_annos,dt_annos,2)
            result_official_dict_2 = get_official_eval_result(
                gt_annos_2, dt_annos_2, self._class_names, z_axis=z_axis, z_center=z_center,difficultys=[0]
            )

            results = {
                "results": {
                    "official": result_official_dict["result"],
                    "official_1": result_official_dict_1["result"],
                    "official_2": result_official_dict_2["result"],
                },
                "detail": {
                    "eval.lvx": {
                        "official": result_official_dict["detail"],
                    }
                },
            }
        else:
            results = None


        return results, dt_annos

    def __getitem__(self, idx):
        return self.get_sensor_data(idx)

    def process(self,idx):
        ''' 
        单帧点云的预处理
        '''
        info = self._lvx_infos[idx]

        res = {
            "lidar": {
                "type": "lidar",
                "points": None,
            },
            "metadata": {
                "image_prefix": self._root_path,
                "num_point_features": LvxDataset.NumPointFeatures,
                "token":str(info["token"]),
            },
            "mode": "val" if self.test_mode else "train",
        }

        data, _ = self.pipeline(res, info)
        return data

    def get_sensor_data(self, idx):

        # 取当前帧及之前两帧的点云
        for i,start in enumerate(self._video_times):
            if idx < start:
                idx = self._start_idx[i-1][0]+2+idx-self._video_times[i-1]

        info = self._lvx_infos[idx]
        info_1 = self._lvx_infos[idx-1]
        info_2 = self._lvx_infos[idx-2]

        res = {
            "lidar": {
                "type": "lidar",
                "points": None,
            },
            "metadata": {
                "image_prefix": self._root_path,
                "num_point_features": LvxDataset.NumPointFeatures,
                "token":str(info["token"]),
            },
            "mode": "val" if self.test_mode else "train",
        }
        res_1 = {
            "lidar": {
                "type": "lidar",
                "points": None,
            },
            "metadata": {
                "image_prefix": self._root_path,
                "num_point_features": LvxDataset.NumPointFeatures,
                "token":str(info["token"]),
            },
            "mode": "val" if self.test_mode else "train",
        }
        res_2 = {
            "lidar": {
                "type": "lidar",
                "points": None,
            },
            "metadata": {
                "image_prefix": self._root_path,
                "num_point_features": LvxDataset.NumPointFeatures,
                "token":str(info["token"]),
            },
            "mode": "val" if self.test_mode else "train",
        }
        # 网络的输入为 3*data

        for t in self.pipeline.transforms:
            if 'shuffle_points' in dir(t):
                # 说明为preprocess类
                res, info, res_1, info_1,res_2, info_2 = t(res, info, res_1, info_1,res_2, info_2)
            else:
                res, info = t(res, info)
                res_1, info_1 = t(res_1, info_1)
                res_2, info_2 = t(res_2, info_2)
            if res is None:
                return None

        data = res
        
        data[f'voxels_1'] = res_1['voxels']
        data[f'coordinates_1'] = res_1['coordinates']
        data[f'num_points_1'] = res_1['num_points']
        data[f'num_voxels_1'] = res_1['num_voxels']
        data[f'voxels_2'] = res_2['voxels']
        data[f'coordinates_2'] = res_2['coordinates']
        data[f'num_points_2'] = res_2['num_points']
        data[f'num_voxels_2'] = res_2['num_voxels']

        return data
