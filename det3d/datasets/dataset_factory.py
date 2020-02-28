from .kitti import KittiDataset
from .nuscenes import NuScenesDataset
from .lyft import LyftDataset
from .lvx import LvxDataset

dataset_factory = {
    "KITTI": KittiDataset,
    "NUSC": NuScenesDataset,
    "LYFT": LyftDataset,
    "LVX": LvxDataset,
}


def get_dataset(dataset_name):
    return dataset_factory[dataset_name]
