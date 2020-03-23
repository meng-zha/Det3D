# TODO: lvx dataset 类
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


@DATASETS.register_module
class LvxDataset(PointCloudDataset):

    NumPointFeatures = 3
    # NumPointFeatures = 6

    def __init__(
        self,
        root_path,
        info_path,
        cfg=None,
        pipeline=None,
        class_names=None,
        test_mode=False,
        **kwargs
    ):
        super(LvxDataset, self).__init__(
            root_path, info_path, pipeline, test_mode=test_mode
        )
        assert self._info_path is not None
        if not hasattr(self, "_lvx_infos"):
            self._lvx_infos = self.load_infos(self._info_path)
        self._num_point_features = __class__.NumPointFeatures
        # print("remain number of infos:", len(self._lvx_infos))
        self._class_names = class_names
    
    def load_infos(self,info_path):
        
        with open(self._info_path,"rb") as f:
            return pickle.load(f)

    def __len__(self):

        if not hasattr(self, "_lvx_infos"):
            self._lvx_infos = self.load_infos(self._info_path)

        return len(self._lvx_infos)

    @property
    def num_point_features(self):
        return self._num_point_features

    @property
    def ground_truth_annotations(self):
        if "annos" not in self._lvx_infos[0]:
            return None

        gt_annos = [info["annos"] for info in self._lvx_infos]

        return gt_annos

    def convert_detection_to_lvx_annos(self, detection):
        class_names = self._class_names
        # det_image_idxes = [k for k in detection.keys()]
        gt_image_idxes = [str(info["token"]) for info in self._lvx_infos]
        # print(f"det_image_idxes: {det_image_idxes[:10]}")
        # print(f"gt_image_idxes: {gt_image_idxes[:10]}")
        annos = []
        # for det_idx in range(len(detection)):
        for det_idx in gt_image_idxes:
            det = detection[det_idx]
            info = self._lvx_infos[gt_image_idxes.index(det_idx)]
            # info = self._lvx_infos[det_idx]
            final_box_preds = det["box3d_lidar"].detach().cpu().numpy()
            label_preds = det["label_preds"].detach().cpu().numpy()
            scores = det["scores"].detach().cpu().numpy()

            anno = get_start_result_anno()
            num_example = 0

            if final_box_preds.shape[0] != 0:
                final_box_preds[:, -1] = box_np_ops.limit_period(
                    final_box_preds[:, -1], offset=0.5, period=np.pi * 2,
                )
                bbox = np.asarray([0,0,500,500])
                for j in range(final_box_preds.shape[0]):
                    anno["bbox"].append(bbox)
                    anno["alpha"].append(-10)
                    anno["dimensions"].append(final_box_preds[j, 3:6])
                    anno["location"].append(final_box_preds[j, :3])
                    anno["rotation_y"].append(final_box_preds[j, 6])
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

    def evaluation(self, detections, output_dir=None, vis = False):
        """
        detection
        When you want to eval your own dataset, you MUST set correct
        the z axis and box z center.
        """
        gt_annos = self.ground_truth_annotations
        dt_annos = self.convert_detection_to_lvx_annos(detections)

        if vis:
            lvx_vis(gt_annos,dt_annos,output_dir)

        # firstly convert standard detection to lvx-format dt annos
        z_axis = 2  # KITTI camera format use y as regular "z" axis.
        z_center = 0.5  # KITTI camera box's center is [0.5, 1, 0.5]
        # for regular raw lidar data, z_axis = 2, z_center = 0.5.

        result_official_dict = get_official_eval_result(
            gt_annos, dt_annos, self._class_names, z_axis=z_axis, z_center=z_center,difficultys=[0]
        )

        results = {
            "results": {
                "official": result_official_dict["result"],
            },
            "detail": {
                "eval.lvx": {
                    "official": result_official_dict["detail"],
                }
            },
        }

        return results, dt_annos

    def __getitem__(self, idx):
        return self.get_sensor_data(idx)

    def get_sensor_data(self, idx):

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
            "calib": None,
            "mode": "val" if self.test_mode else "train",
        }

        data, _ = self.pipeline(res, info)

        return data
