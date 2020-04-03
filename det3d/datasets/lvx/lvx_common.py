import pathlib
import pickle
import re
import numpy as np
import open3d as o3d

from collections import OrderedDict
from pathlib import Path
from skimage import io
from tqdm import tqdm

from det3d.core import box_np_ops


def _read_imageset_file(path):
    with open(path, "r") as f:
        lines = f.readlines()
    return [int(line) for line in lines]


def _calculate_num_points_in_gt(
    data_path, infos, relative_path, remove_outside=False, num_features=3
):
    '''
    计算bounding box内点的数量
    '''
    for info in infos:
        pc_info = info["point_cloud"]
        if relative_path:
            v_path = str(Path(data_path) / pc_info["velodyne_path"])
        else:
            v_path = pc_info["velodyne_path"]
        pcd = o3d.io.read_point_cloud(str(v_path))
        points_v = np.asarray(pcd.points)
        
        if num_features >= 4:
            normals_v = np.asarray(pcd.normals)
            points_v = np.concatenate([points_v,normals_v],axis=1)[:,:num_features]

        annos = info["annos"]
        dims = annos["dimensions"]
        loc = annos["location"]
        rots = annos["rotation_y"]
        gt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1)
        indices = box_np_ops.points_in_rbbox(points_v[:, :3], gt_boxes)
        num_points_in_gt = indices.sum(0)
        annos["num_points_in_gt"] = num_points_in_gt 


def create_lvx_info_file(data_path, save_path=None, relative_path=True):
    imageset_folder = Path(__file__).resolve().parent.parent / "ImageSets"
    train_img_ids = _read_imageset_file(str(imageset_folder / "train_lvx.txt"))
    val_img_ids = _read_imageset_file(str(imageset_folder / "val_lvx.txt"))
    test_img_ids = _read_imageset_file(str(imageset_folder / "test_lvx.txt"))

    print("Generate info. this may take several minutes.")
    if save_path is None:
        save_path = Path(data_path)
    else:
        save_path = Path(save_path)

    lvx_infos_train = get_lvx_image_info(
        data_path,
        training=True,
        lidar=True,
        image_ids=train_img_ids,
        relative_path=relative_path,
    )
    _calculate_num_points_in_gt(data_path, lvx_infos_train, relative_path)
    filename = save_path / "lvx_infos_train.pkl"
    print(f"Lvx info train file is saved to {filename}")
    with open(filename, "wb") as f:
        pickle.dump(lvx_infos_train, f)
    lvx_infos_val = get_lvx_image_info(data_path,
                                           training=True,
                                           lidar=True,
                                           image_ids=val_img_ids,
                                           relative_path=relative_path)
    _calculate_num_points_in_gt(data_path, lvx_infos_val, relative_path)
    filename = save_path / 'lvx_infos_val.pkl'
    print(f"Lvx info val file is saved to {filename}")
    with open(filename, 'wb') as f:
        pickle.dump(lvx_infos_val, f)
    filename = save_path / 'lvx_infos_trainval.pkl'
    print(f"Lvx info trainval file is saved to {filename}")
    with open(filename, 'wb') as f:
        pickle.dump(lvx_infos_train + lvx_infos_val, f)
    lvx_infos_test = get_lvx_image_info(data_path,
                                                  training=False,
                                                  label_info=False,
                                                  lidar=True,
                                                  image_ids=test_img_ids,
                                                  relative_path=relative_path)
    filename = save_path / 'lvx_infos_test.pkl'
    print(f"lvx info test file is saved to {filename}")
    with open(filename, 'wb') as f:
        pickle.dump(lvx_infos_test, f)


def area(boxes, add1=False):
    """Computes area of boxes.

    Args:
        boxes: Numpy array with shape [N, 4] holding N boxes

    Returns:
        a numpy array with shape [N*1] representing box areas
    """
    if add1:
        return (boxes[:, 2] - boxes[:, 0] + 1.0) * (boxes[:, 3] - boxes[:, 1] + 1.0)
    else:
        return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def intersection(boxes1, boxes2, add1=False):
    """Compute pairwise intersection areas between boxes.

    Args:
        boxes1: a numpy array with shape [N, 4] holding N boxes
        boxes2: a numpy array with shape [M, 4] holding M boxes

    Returns:
        a numpy array with shape [N*M] representing pairwise intersection area
    """
    [y_min1, x_min1, y_max1, x_max1] = np.split(boxes1, 4, axis=1)
    [y_min2, x_min2, y_max2, x_max2] = np.split(boxes2, 4, axis=1)

    all_pairs_min_ymax = np.minimum(y_max1, np.transpose(y_max2))
    all_pairs_max_ymin = np.maximum(y_min1, np.transpose(y_min2))
    if add1:
        all_pairs_min_ymax += 1.0
    intersect_heights = np.maximum(
        np.zeros(all_pairs_max_ymin.shape), all_pairs_min_ymax - all_pairs_max_ymin
    )

    all_pairs_min_xmax = np.minimum(x_max1, np.transpose(x_max2))
    all_pairs_max_xmin = np.maximum(x_min1, np.transpose(x_min2))
    if add1:
        all_pairs_min_xmax += 1.0
    intersect_widths = np.maximum(
        np.zeros(all_pairs_max_xmin.shape), all_pairs_min_xmax - all_pairs_max_xmin
    )
    return intersect_heights * intersect_widths


def iou(boxes1, boxes2, add1=False):
    """Computes pairwise intersection-over-union between box collections.

    Args:
        boxes1: a numpy array with shape [N, 4] holding N boxes.
        boxes2: a numpy array with shape [M, 4] holding N boxes.

    Returns:
        a numpy array with shape [N, M] representing pairwise iou scores.
    """
    intersect = intersection(boxes1, boxes2, add1)
    area1 = area(boxes1, add1)
    area2 = area(boxes2, add1)
    union = np.expand_dims(area1, axis=1) + np.expand_dims(area2, axis=0) - intersect
    return intersect / union


def get_image_index_str(img_idx):
    return f"PC_{img_idx}"


def get_lvx_info_path(
    idx,
    prefix,
    info_type="Lidar",
    file_tail=".pcd",
    training=True,
    relative_path=True,
    exist_check=True,
):
    img_idx_str = get_image_index_str(idx)
    img_idx_str += file_tail
    prefix = pathlib.Path(prefix)
    if training:
        file_path = pathlib.Path("training") / info_type / img_idx_str
    else:
        file_path = pathlib.Path("testing") / info_type / img_idx_str
    if exist_check and not (prefix / file_path).exists():
        raise ValueError("file not exist: {}".format(file_path))
    if relative_path:
        return str(file_path)
    else:
        return str(prefix / file_path)


def get_label_path(idx, prefix, training=True, relative_path=True, exist_check=True):
    return get_lvx_info_path(
        idx, prefix, "Label", ".txt", training, relative_path, exist_check
    )


def get_lidar_path(idx, prefix, training=True, relative_path=True, exist_check=True):
    return get_lvx_info_path(
        idx, prefix, "Lidar", ".pcd", training, relative_path, exist_check
    )


def get_lvx_image_info(
    path,
    training=True,
    label_info=True,
    lidar=False,
    calib=False,
    image_ids=400,
    extend_matrix=True,
    num_worker=8,
    relative_path=True,
):
    # image_infos = []
    """
    KITTI annotation format version 2:
    {
        point_cloud: {
            num_features: 4
            velodyne_path: ...
        }
        annos: {
            location: [num_gt, 3] array
            dimensions: [num_gt, 3] array
            rotation_y: [num_gt] angle array
            name: [num_gt] ground truth name array
            [optional]difficulty: kitti difficulty
            [optional]group_ids: used for multi-part object
        }
    }
    """
    root_path = pathlib.Path(path)
    if not isinstance(image_ids, list):
        image_ids = list(range(image_ids))

    def map_func(idx):
        info = {}
        pc_info = {"num_features": 3} 
        calib_info = {}

        annotations = None
        if lidar:
            pc_info["velodyne_path"] = get_lidar_path(
                idx, path, training, relative_path
            )
        if label_info:
            label_path = get_label_path(idx, path, training, relative_path)
            if relative_path:
                label_path = str(root_path / label_path)
            annotations = get_label_anno(label_path,idx)
        info["point_cloud"] = pc_info
        info["token"]=idx

        if annotations is not None:
            info["annos"] = annotations
        return info

    image_infos = []
    for i in tqdm(image_ids):
        image_infos.append(map_func(i))

    return image_infos


def get_class_to_label_map():
    class_to_label = {
        "Car": 0,
        "Pedestrian": 1,
        "Cyclist": 2,
        "Van": 3,
        "Person_sitting": 4,
        "Truck": 5,
        "Tram": 6,
        "Misc": 7,
        "DontCare": -1,
    }
    return class_to_label


def get_classes():
    return get_class_to_label_map().keys()

def get_label_anno(label_path,idx):
    '''
    构建label，其中dim,loc,rot应该有3帧的数据
    return {name": [],
            "truncated": [],
            "occluded": [],
            "alpha": [],
            "bbox": [],
            "dimensions": [],
            "location": [],
            "rotation_y": [],
            "dimensions_1": [],
            "location_1": [],
            "rotation_y_1": [],
            "dimensions_2": [],
            "location_2": [],
            "rotation_y_2": [],
            "token":idx,}
    '''
    annotations = {}
    annotations.update(
        {
            "name": [],
            "truncated": [],
            "occluded": [],
            "alpha": [],
            "bbox": [],
            "dimensions": [],
            "location": [],
            "rotation_y": [],
            "token":idx,
        }
    )
    with open(label_path, "r") as f:
        lines = f.readlines()
    content = [line.strip().split(" ") for line in lines]

    name = ['','Pedestrian','DontCare']
    num_objects = len([x[0] for x in content if float(x[0]) != -1])
    annotations["name"] = np.array([name[int(float(x[0]))] for x in content])
    num_gt = len(annotations["name"])
    annotations["truncated"] = np.zeros((num_gt,))
    annotations["occluded"] = np.zeros((num_gt,))
    annotations["alpha"] = -10*np.ones((num_gt,))
    # fake bbox lack of images
    annotations["bbox"] = np.array(
        [[0,0,500,500] for x in content]
    ).reshape(-1, 4)
    # dimensions will convert wlh format to standard lwh(lvx) format.
    annotations["dimensions"] = np.array(
        [[float(x[2]),float(x[1]),float(x[3])] for x in content]
    ).reshape(-1, 3)
    annotations["location"] = np.array(
        [[float(info) for info in x[4:7]] for x in content]
    ).reshape(-1, 3)
    annotations["rotation_y"] = -1*np.array([float(x[7]) for x in content]).reshape(-1)
    # 之前一帧的对应标注
    annotations["dimensions_1"] = np.array(
        [[float(x[2+9]),float(x[1+9]),float(x[3+9])] for x in content]
    ).reshape(-1, 3)
    annotations["location_1"] = np.array(
        [[float(info) for info in x[(4+9):(7+9)]] for x in content]
    ).reshape(-1, 3)
    annotations["rotation_y_1"] = -1*np.array([float(x[7+9]) for x in content]).reshape(-1)

    annotations["dimensions_2"] = np.array(
        [[float(x[2+9*2]),float(x[1+9*2]),float(x[3+9*2])] for x in content]
    ).reshape(-1, 3)
    annotations["location_2"] = np.array(
        [[float(info) for info in x[(4+9*2):(7+9*2)]] for x in content]
    ).reshape(-1, 3)
    annotations["rotation_y_2"] = -1*np.array([float(x[7+9*2]) for x in content]).reshape(-1)

    # T-2帧的label，用于gtbox 的 gtaug， T-2帧的点需要对应旋转缩放
    annotations["dimensions_3"] = np.array(
        [[float(x[2+9*3]),float(x[1+9*3]),float(x[3+9*3])] for x in content]
    ).reshape(-1, 3)
    annotations["location_3"] = np.array(
        [[float(info) for info in x[(4+9*3):(7+9*3)]] for x in content]
    ).reshape(-1, 3)
    annotations["rotation_y_3"] = -1*np.array([float(x[7+9*3]) for x in content]).reshape(-1)
    index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
    annotations["group_ids"] = np.arange(num_gt, dtype=np.int32)
    return annotations


def get_start_result_anno():
    annotations = {}
    annotations.update(
        {
            # 'index': None,
            "name": [],
            "truncated": [],
            "occluded": [],
            "alpha": [],
            "bbox": [],
            "dimensions": [],
            "location": [],
            "rotation_y": [],
            "dimensions_1": [],
            "location_1": [],
            "rotation_y_1": [],
            "dimensions_2": [],
            "location_2": [],
            "rotation_y_2": [],
            "score": [],
        }
    )
    return annotations


def empty_result_anno():
    annotations = {}
    annotations.update(
        {
            "name": np.array([]),
            "truncated": np.array([]),
            "occluded": np.array([]),
            "alpha": np.array([]),
            "bbox": np.zeros([0, 4]),
            "dimensions": np.zeros([0, 3]),
            "location": np.zeros([0, 3]),
            "rotation_y": np.array([]),
            "dimensions_1": np.zeros([0, 3]),
            "location_1": np.zeros([0, 3]),
            "rotation_y_1": np.array([]),
            "dimensions_2": np.zeros([0, 3]),
            "location_2": np.zeros([0, 3]),
            "rotation_y_2": np.array([]),
            "dimensions_3": np.zeros([0, 3]),
            "location_3": np.zeros([0, 3]),
            "rotation_y_3": np.array([]),
            "score": np.array([]),
        }
    )
    return annotations


def get_label_annos(label_folder, image_ids=None):
    if image_ids is None:
        filepaths = pathlib.Path(label_folder).glob("*.txt")
        prog = re.compile(r"^\d{6}.txt$")
        filepaths = filter(lambda f: prog.match(f.name), filepaths)
        image_ids = [int(p.stem) for p in filepaths]
        image_ids = sorted(image_ids)
    if not isinstance(image_ids, list):
        image_ids = list(range(image_ids))
    annos = []
    label_folder = pathlib.Path(label_folder)
    for idx in image_ids:
        image_idx_str = get_image_index_str(idx)
        label_filename = label_folder / (image_idx_str + ".txt")
        anno = get_label_anno(label_filename,idx)
        num_example = anno["name"].shape[0]
        anno["image_idx"] = np.array([idx] * num_example, dtype=np.int64)
        annos.append(anno)
    return annos


def anno_to_rbboxes(anno):
    loc = anno["location"]
    dims = anno["dimensions"]
    rots = anno["rotation_y"]
    rbboxes = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1)
    return rbboxes
