from ..registry import DETECTORS
from .single_stage import SingleStageDetector


@DETECTORS.register_module
class PointPillars(SingleStageDetector):
    def __init__(
        self,
        reader,
        backbone,
        neck,
        bbox_head,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
    ):
        super(PointPillars, self).__init__(
            reader, backbone, neck, bbox_head, train_cfg, test_cfg, pretrained
        )

    def extract_feat(self, data,data_1,data_2):
        input_features = self.reader(
            data["features"], data["num_voxels"], data["coors"]
        )
        x = self.backbone(
            input_features, data["coors"], data["batch_size"], data["input_shape"]
        )
        input_features = self.reader(
            data_1["features"], data_1["num_voxels"], data_1["coors"]
        )
        x_1 = self.backbone(
            input_features, data_1["coors"], data_1["batch_size"], data_1["input_shape"]
        )
        input_features = self.reader(
            data_2["features"], data_2["num_voxels"], data_2["coors"]
        )
        x_2 = self.backbone(
            input_features, data_2["coors"], data_2["batch_size"], data_2["input_shape"]
        )
        if self.with_neck:
            x = self.neck(x,x_1,x_2)
        return x

    def forward(self, example, return_loss=True, **kwargs):
        def extract(example,idx):
            '''
            在example中重复3次 pfn网络，提取时域的特征
            '''
            voxels = example["voxels"]
            coordinates = example["coordinates"]
            num_points_in_voxel = example["num_points"]
            num_voxels = example["num_voxels"]

            if idx == 1 or idx ==2:
                voxels = example[f"voxels_{idx}"]
                coordinates = example[f"coordinates_{idx}"]
                num_points_in_voxel = example[f"num_points_{idx}"]
                num_voxels = example[f"num_voxels_{idx}"]

            batch_size = len(num_voxels)

            data = dict(
                features=voxels,
                num_voxels=num_points_in_voxel,
                coors=coordinates,
                batch_size=batch_size,
                input_shape=example["shape"][0],
            )

            # x = self.extract_feat(data)
            # return x
            return data
        time_series0 = extract(example,0)
        time_series1 = extract(example,1)
        time_series2 = extract(example,2)

        x = self.extract_feat(time_series0,time_series1,time_series2)

        # import torch
        # x = torch.cat([time_series0,time_series1,time_series2],dim=1)

        preds = self.bbox_head(x)

        if return_loss:
            return self.bbox_head.loss(example, preds)
        else:
            return self.bbox_head.predict(example, preds, self.test_cfg)
