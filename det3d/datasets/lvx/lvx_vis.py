import os
import numpy as np
from OpenGL.GL import glLineWidth
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import argparse
import open3d as o3d
import mayavi.mlab as mlab
mlab.options.offscreen = True
import cv2

from det3d.datasets.utils.eval import calculate_iou_partly
from det3d.core.bbox.box_np_ops import center_to_corner_box3d
from .viz_util import draw_gt_boxes3d,draw_lidar
camera = [-106.4126,46.4585,21.0723]

class Object3d(object):
    """ 3d object label """

    def __init__(self, annos, idx):
        # extract label, truncation, occlusion
        self.type = annos["name"][idx]  # 'Car', 'Pedestrian', ...
        # self.truncation = annos["truncated"][idx]  # truncated pixel ratio [0..1]
        # self.occlusion = int(
        #     annos["occluded"][idx]
        # )  # 0=visible, 1=partly occluded, 2=fully occluded, 3=unknown
        self.alpha = annos["alpha"][idx]  # object observation angle [-pi..pi]

        # extract 2d bounding box in 0-based coordinates
        self.xmin = annos["bbox"][idx][0]  # left
        self.ymin = annos["bbox"][idx][1]  # top
        self.xmax = annos["bbox"][idx][2]  # right
        self.ymax = annos["bbox"][idx][3]  # bottom
        self.box2d = annos["bbox"][idx]

        # extract 3d bounding box information
        self.h = annos["dimensions"][idx][2]  # box height
        self.w = annos["dimensions"][idx][1]  # box width
        self.l = annos["dimensions"][idx][0]  # box length (in meters)
        self.h_1 = annos["dimensions_1"][idx][2]  # box height
        self.w_1 = annos["dimensions_1"][idx][1]  # box width
        self.l_1 = annos["dimensions_1"][idx][0]  # box length (in meters)
        self.h_2 = annos["dimensions_2"][idx][2]  # box height
        self.w_2 = annos["dimensions_2"][idx][1]  # box width
        self.l_2 = annos["dimensions_2"][idx][0]  # box length (in meters)
        # location (x,y,z) in camera coord.
        self.t = list(annos["location"][idx])
        self.t_1 = list(annos["location_1"][idx])
        self.t_2 = list(annos["location_2"][idx])
        self.rz = annos["rotation_y"][idx]  # yaw angle (around Y-axis in camera coordinates) [-pi..pi]
        self.rz_1 = annos["rotation_y_1"][idx]  # yaw angle (around Y-axis in camera coordinates) [-pi..pi]
        self.rz_2 = annos["rotation_y_2"][idx]  # yaw angle (around Y-axis in camera coordinates) [-pi..pi]

        self.track = -1
        if "track_id" in annos.keys():
            self.track = annos["track_id"][idx]
        self.id = -1
        if "id" in annos.keys():
            self.id = annos["id"][idx]

    def print_object(self):
        print(
            "Type, truncation, occlusion, alpha: %s, %d, %d, %f"
            % (self.type, self.truncation, self.occlusion, self.alpha)
        )
        print(
            "2d bbox (x0,y0,x1,y1): %f, %f, %f, %f"
            % (self.xmin, self.ymin, self.xmax, self.ymax)
        )
        print("3d bbox h,w,l: %f, %f, %f" % (self.h, self.w, self.l))
        print(
            "3d bbox location, rz: (%f, %f, %f), %f"
            % (self.t[0], self.t[1], self.t[2], self.rz)
        )


# -----------------------------------------------------------------------------------------


def inverse_rigid_trans(Tr):
    """ Inverse a rigid body transform matrix (3x4 as [R|t])
        [R'|-R't; 0|1]
    """
    inv_Tr = np.zeros_like(Tr)  # 3x4
    inv_Tr[0:3, 0:3] = np.transpose(Tr[0:3, 0:3])
    inv_Tr[0:3, 3] = np.dot(-np.transpose(Tr[0:3, 0:3]), Tr[0:3, 3])
    return inv_Tr

# -----------------------------------------------------------------------------------------

def load_velo_scan(velo_filename):
    pcd = o3d.io.read_point_cloud(velo_filename)
    scan = np.asarray(pcd.points)
    norm = np.asarray(pcd.normals)
    if norm.shape[0] > 0:
        return np.concatenate([scan,norm],axis=1)[:,:4]
    else:
        return scan


def read_label(label_filename):
    lines = [line.rstrip() for line in open(label_filename)]
    objects = [Object3d(line) for line in lines]
    return objects


class lvx_object(object):
    """Load and parse object data into a usable format."""

    def __init__(self, root_dir,split='training'):
        """root_dir contains training and testing folders"""
        self.root_dir = root_dir
        self.split=split

        self.lidar_dir = os.path.join(self.root_dir, self.split, "Lidar")
        # self.lidar_dir = '/Extra/zhangmeng/3d_detection/zm_dataset/pc_out/samplex4'
        self.label_dir = os.path.join(self.root_dir, self.split, "Label")

    def get_lidar(self, idx):
        lidar_filename = os.path.join(self.lidar_dir, f"PC_{idx}.pcd")
        return load_velo_scan(lidar_filename)

    def get_label_objects(self, gt_annos):
        return [Object3d(gt_annos,idx) for idx in range(len(gt_annos["name"]))]


# -----------------------------------------------------------------------------------------


def rotx(t):
    """ 3D Rotation about the x-axis. """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def roty(t):
    """ Rotation about the y-axis. """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rotz(t):
    """ Rotation about the z-axis. """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])


def compute_box_3d(obj):
    """ Takes an object and a projection matrix (P) and projects the 3d
        bounding box into the image plane.
        Returns:
            corners_2d: (8,2) array in left image coord.
            corners_3d: (8,3) array in in rect camera coord.
    """
    # compute rotational matrix around yaw axis
    R = rotz(obj.rz)
    # 3d bounding box dimensions
    l = obj.l
    w = obj.w
    h = obj.h
    # 3d bounding box corners
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    z_corners = [0, 0, 0, 0, -h, -h, -h, -h]+h/2
    y_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    # rotate and translate 3d bounding box
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    # print corners_3d.shape
    corners_3d[0, :] = corners_3d[0, :] + obj.t[0]
    corners_3d[1, :] = corners_3d[1, :] + obj.t[1]
    corners_3d[2, :] = corners_3d[2, :] + obj.t[2]
    # print 'cornsers_3d: ', corners_3d

    return np.transpose(corners_3d)

def compute_box_3d_track(l,w,h,rz,x,y,z):
    """ Takes an object and a projection matrix (P) and projects the 3d
        bounding box into the image plane.
        Returns:
            corners_2d: (8,2) array in left image coord.
            corners_3d: (8,3) array in in rect camera coord.
    """
    # compute rotational matrix around yaw axis
    R = rotz(rz)
    # 3d bounding box dimensions
    # 3d bounding box corners
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    z_corners = [0, 0, 0, 0, -h, -h, -h, -h]+h/2
    y_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    # rotate and translate 3d bounding box
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    # print corners_3d.shape
    corners_3d[0, :] = corners_3d[0, :] + x
    corners_3d[1, :] = corners_3d[1, :] + y
    corners_3d[2, :] = corners_3d[2, :] + z
    # print 'cornsers_3d: ', corners_3d

    return np.transpose(corners_3d)

def compute_orientation_3d(obj):
    """ Takes an object and a projection matrix (P) and projects the 3d
        object orientation vector into the image plane.
        Returns:
            orientation_2d: (2,2) array in left image coord.
            orientation_3d: (2,3) array in in rect camera coord.
    """
    # compute rotational matrix around yaw axis
    R = rotz(obj.rz)
    # orientation in object coordinate system
    orientation_3d = np.array([[0.0, 3*obj.l], [0.0, 0], [0, 0]])
    # rotate and translate in camera coordinate system, project in image
    orientation_3d = np.dot(R, orientation_3d)
    orientation_3d[0, :] = orientation_3d[0, :] + obj.t[0]
    orientation_3d[1, :] = orientation_3d[1, :] + obj.t[1]
    orientation_3d[2, :] = orientation_3d[2, :] + obj.t[2]
    return np.transpose(orientation_3d)


# -----------------------------------------------------------------------------------------


def create_bbox_mesh(p3d, gt_boxes3d):
    b = gt_boxes3d
    for k in range(0, 4):
        i, j = k, (k + 1) % 4
        p3d.add_line([b[i, 0], b[i, 1], b[i, 2]], [b[j, 0], b[j, 1], b[j, 2]])
        i, j = k + 4, (k + 1) % 4 + 4
        p3d.add_line([b[i, 0], b[i, 1], b[i, 2]], [b[j, 0], b[j, 1], b[j, 2]])
        i, j = k, k + 4
        p3d.add_line([b[i, 0], b[i, 1], b[i, 2]], [b[j, 0], b[j, 1], b[j, 2]])


class plot3d(object):
    def __init__(self):
        self.app = pg.mkQApp()
        self.view = gl.GLViewWidget()
        coord = gl.GLAxisItem()
        glLineWidth(3)
        coord.setSize(3, 3, 3)
        self.view.addItem(coord)

    def add_points(self, points, colors):
        points_item = gl.GLScatterPlotItem(pos=points, size=2, color=colors)
        self.view.addItem(points_item)

    def add_line(self, p1, p2):
        lines = np.array([[p1[0], p1[1], p1[2]], [p2[0], p2[1], p2[2]]])
        lines_item = gl.GLLinePlotItem(
            pos=lines, mode="lines", color=(1, 0, 0, 1), width=3, antialias=True
        )
        self.view.addItem(lines_item)

    def show(self):
        self.view.show()
        self.app.exec()


def show_lidar_with_boxes(pc_velo, objects, calib):
    p3d = plot3d()
    points = pc_velo[:, 0:3]
    pc_inte = pc_velo[:, 3]
    pc_color = inte_to_rgb(pc_inte)
    p3d.add_points(points, pc_color)
    for obj in objects:
        if obj.type == "DontCare":
            continue
        # Draw 3d bounding box
        box3d_pts_3d = compute_box_3d(obj)
        create_bbox_mesh(p3d, box3d_pts_3d)
        # Draw heading arrow
        ori3d_pts_3d = compute_orientation_3d(obj)
        x1, y1, z1 = ori3d_pts_3d[0, :]
        x2, y2, z2 = ori3d_pts_3d[1, :]
        p3d.add_line([x1, y1, z1], [x2, y2, z2])
    p3d.show()

def show_trajectory(gt_objects,dt_objects,gt_plt,dt_plt,overlap,idx):
    obj = gt_objects[idx]
    box3d_pts_3d = compute_box_3d_track(obj.l,obj.w,obj.h,obj.rz,*obj.t)
    for k in range(4):
        i, j = k, (k + 1) % 4
        gt_plt.plot([box3d_pts_3d[i,0],box3d_pts_3d[j,0]],[box3d_pts_3d[i,1],box3d_pts_3d[j,1]],color='r',linewidth=0.3)

    track_id = -1
    if np.array(overlap[idx]).max() > 0.2:
        track_id = np.array(overlap[gt_objects.index(obj)]).argmax()
    
    if track_id != -1:
        obj = dt_objects[track_id]
        box3d_pts_3d = compute_box_3d_track(obj.l,obj.w,obj.h,obj.rz,*obj.t)
        track = obj.track
        np.random.seed(int(abs(track)))
        c = np.random.rand(3)
        for k in range(4):
            i, j = k, (k + 1) % 4
            dt_plt.plot([box3d_pts_3d[i,0],box3d_pts_3d[j,0]],[box3d_pts_3d[i,1],box3d_pts_3d[j,1]],color=c,linewidth=0.3)



def show_bev_objects(lidar,lidar_1,lidar_2,gt_objects,dt_objects,output_dir,overlap = None, miss = None):
    points = lidar[:,:4]
    color_map = ['r','g','b','y']
    # colors = [color_map[int(i)] for i in points[:,3]]
    plt.clf()
    # plt.scatter(points[:,0],points[:,1],c=colors ,s=0.1*np.ones((len(points[:,2]),)),linewidth=0.1*np.ones((len(points[:,2]),)))
    plt.scatter(lidar_1[:,0],lidar_1[:,1],c='y',s=0.3*np.ones((len(lidar_1[:,2]),)),linewidth=0.3*np.ones((len(lidar_1[:,2]),)))
    plt.scatter(points[:,0],points[:,1],s=0.3*np.ones((len(points[:,2]),)),linewidth=0.3*np.ones((len(points[:,2]),)))
    if lidar_2 != []:
        plt.scatter(lidar_2[:,0],lidar_2[:,1],c='g',s=0.3*np.ones((len(lidar_2[:,2]),)),linewidth=0.3*np.ones((len(lidar_2[:,2]),)))


    for obj in gt_objects:
        if obj.t[0]<-40 or obj.t[0]>40 or obj.t[1]<-40 or obj.t[1]>40:
            miss[obj.id] = -2
        
        if np.array(overlap[gt_objects.index(obj)]).max() > 0.25:
            miss[obj.id] = dt_objects[np.array(overlap[gt_objects.index(obj)]).argmax()].track

        if obj.type == "DontCare":
            # 如果没有点，设为标志-1
            dist = (((np.array(obj.t)-camera)**2).sum())**0.5
            miss[obj.id] = -1
            continue

        if np.array(overlap[gt_objects.index(obj)]).max() < 0.25:
            # 如果有点，但是没有检测出来，设为标志1
            dist = (((np.array(obj.t)-camera)**2).sum())**0.5
            miss[obj.id]=-1
        # Draw bev bounding box
        box3d_pts_3d = compute_box_3d_track(obj.l,obj.w,obj.h,obj.rz,*obj.t)
        for k in range(4):
            i, j = k, (k + 1) % 4
            plt.plot([box3d_pts_3d[i,0],box3d_pts_3d[j,0]],[box3d_pts_3d[i,1],box3d_pts_3d[j,1]],color='r',linewidth=0.3)

        # box3d_pts_3d_1 = compute_box_3d_track(obj.l_1,obj.w_1,obj.h_1,obj.rz_1,*obj.t_1)
        # for k in range(4):
        #     i, j = k, (k + 1) % 4
        #     plt.plot([box3d_pts_3d_1[i,0],box3d_pts_3d_1[j,0]],[box3d_pts_3d_1[i,1],box3d_pts_3d_1[j,1]],color='y',linewidth=0.3)
        # box3d_pts_3d_2 = compute_box_3d_track(obj.l_2,obj.w_2,obj.h_2,obj.rz_2,*obj.t_2)
        # for k in range(4):
        #     i, j = k, (k + 1) % 4
        #     plt.plot([box3d_pts_3d_2[i,0],box3d_pts_3d_2[j,0]],[box3d_pts_3d_2[i,1],box3d_pts_3d_2[j,1]],color='black',linewidth=0.3)
        # Draw heading arrow
        ori3d_pts_3d = compute_orientation_3d(obj)
        x1, y1, z1 = ori3d_pts_3d[0, :]
        x2, y2, z2 = ori3d_pts_3d[1, :]
        plt.text(x1,y1,f"h={obj.h:.1f},z={obj.t[2]:.1f}",c='y',fontsize=3)
        plt.plot([x1, x2], [y1, y2],color='y',linewidth=0.3)

    for obj in dt_objects:
        if obj.type == "DontCare":
            continue
        # Draw bev bounding box
        box3d_pts_3d = compute_box_3d_track(obj.l,obj.w,obj.h,obj.rz,*obj.t)
        np.random.seed(int(abs(obj.track)))
        c = np.random.rand(3)
        for k in range(4):
            i, j = k, (k + 1) % 4
            plt.plot([box3d_pts_3d[i,0],box3d_pts_3d[j,0]],[box3d_pts_3d[i,1],box3d_pts_3d[j,1]],color=c,linewidth=1)

        box3d_pts_3d_1 = compute_box_3d_track(obj.l_1,obj.w_1,obj.h_1,obj.rz_1,*obj.t_1)
        for k in range(4):
            i, j = k, (k + 1) % 4
            plt.plot([box3d_pts_3d_1[i,0],box3d_pts_3d_1[j,0]],[box3d_pts_3d_1[i,1],box3d_pts_3d_1[j,1]],color='gold',linewidth=0.3)
        box3d_pts_3d_2 = compute_box_3d_track(obj.l_2,obj.w_2,obj.h_2,obj.rz_2,*obj.t_2)
        for k in range(4):
            i, j = k, (k + 1) % 4
            plt.plot([box3d_pts_3d_2[i,0],box3d_pts_3d_2[j,0]],[box3d_pts_3d_2[i,1],box3d_pts_3d_2[j,1]],color='indigo',linewidth=0.3)
        # Draw heading arrow
        ori3d_pts_3d = compute_orientation_3d(obj)
        x1, y1, z1 = ori3d_pts_3d[0, :]
        x2, y2, z2 = ori3d_pts_3d[1, :]
        plt.text(x1,y1,f"track={int(obj.track)},h={obj.h:.1f},z={obj.t[2]:.1f}",c='k',fontsize=3)
        plt.plot([x1, x2], [y1, y2],linewidth=0.3,color='k')

    if gt_objects == []:
        plt.xlim(-20,20)
        plt.ylim(-20,20)
    else:
        plt.xlim(-40,40)
        plt.ylim(-40,40)
    plt.rcParams['figure.dpi'] = 500
    plt.rcParams['savefig.dpi'] = 500
    plt.savefig(output_dir)



def inte_to_rgb(pc_inte):
    minimum, maximum = np.min(pc_inte), np.max(pc_inte)
    ratio = 2 * (pc_inte - minimum) / (maximum - minimum)
    b = np.maximum((1 - ratio), 0)
    r = np.maximum((ratio - 1), 0)
    g = 1 - b - r
    return np.stack([r, g, b, np.ones_like(r)]).transpose()

# -----------------------------------------------------------------------------------------
def track_vis(gt_annos,dt_annos,output_dir, track_id):
    root_path = dt_annos[0]["metadata"]["image_prefix"]
    gt_plt = plt.figure()
    plt.xlim(-40,40)
    plt.ylim(-40,40)
    plt.rcParams['figure.dpi'] = 500
    plt.rcParams['savefig.dpi'] = 500
    gt_ax = gt_plt.add_subplot(1,1,1)
    dt_plt = plt.figure()
    plt.xlim(-40,40)
    plt.ylim(-40,40)
    plt.rcParams['figure.dpi'] = 500
    plt.rcParams['savefig.dpi'] = 500
    dt_ax = dt_plt.add_subplot(1,1,1)
    
    if gt_annos is not None:
        overlap,_,_,_ = calculate_iou_partly(gt_annos,dt_annos,2,50,2,0.5)
        dataset = lvx_object(root_path)
        gt_image_idxes = [str(info["token"]) for info in gt_annos]
        for idx in gt_image_idxes:
            print(idx)
            gt_objects = dataset.get_label_objects(gt_annos[gt_image_idxes.index(idx)])
            dt_objects = dataset.get_label_objects(dt_annos[gt_image_idxes.index(idx)])

            show_trajectory(gt_objects,dt_objects,gt_ax,dt_ax,overlap[gt_image_idxes.index(idx)],track_id)
    
    else:
        raise NotImplemented

    gt_img_path = os.path.join(output_dir,f"gt_tracject_{track_id}.png")
    gt_plt.savefig(gt_img_path)
    dt_img_path = os.path.join(output_dir,f"dt_tracject_{track_id}.png")
    dt_plt.savefig(dt_img_path)

# -----------------------------------------------------------------------------------------
def lvx_vis(gt_annos,dt_annos,output_dir):
    root_path = dt_annos[0]["metadata"]["image_prefix"]
    
    if gt_annos is not None:
        miss = -3*np.ones((22,len(gt_annos)))
        overlap,_,_,_ = calculate_iou_partly(gt_annos,dt_annos,2,50,2,0.5)
        dataset = lvx_object(root_path)
        gt_image_idxes = [str(info["token"]) for info in gt_annos]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        vot = cv2.VideoWriter(os.path.join(output_dir,"bev_video.mp4"),fourcc,10,(3200,2400),True)
        for idx in gt_image_idxes:
            print(idx)
            points = dataset.get_lidar(idx)
            points = points[points[:,1]<40,:]
            points = points[points[:,0]<40,:]
            points = points[points[:,1]>-40,:]
            points = points[points[:,0]>-40,:]
            points = points[points[:,2]>0.1,:]

            points_1 = dataset.get_lidar(f'{int(idx)-1}')
            points_1 = points_1[points_1[:,1]<40,:]
            points_1 = points_1[points_1[:,0]<40,:]
            points_1 = points_1[points_1[:,1]>-40,:]
            points_1 = points_1[points_1[:,0]>-40,:]
            points_1 = points_1[points_1[:,2]>0.1,:]

            if (gt_image_idxes.index(idx) < len(gt_image_idxes)-1):
                points_2 = dataset.get_lidar(f'{int(idx)+1}')
                points_2 = points_2[points_2[:,1]<40,:]
                points_2 = points_2[points_2[:,0]<40,:]
                points_2 = points_2[points_2[:,1]>-40,:]
                points_2 = points_2[points_2[:,0]>-40,:]
                points_2 = points_2[points_2[:,2]>0.1,:]
            else:
                points_2 = []

            gt_objects = dataset.get_label_objects(gt_annos[gt_image_idxes.index(idx)])
            dt_objects = dataset.get_label_objects(dt_annos[gt_image_idxes.index(idx)])

            if not os.path.exists(os.path.join(output_dir,"bev_imgs")):
                os.makedirs(os.path.join(output_dir,"bev_imgs"))

            img_path = os.path.join(output_dir,f"bev_imgs/lvx_{idx}.png")

            show_bev_objects(points,points_1,points_2,gt_objects,dt_objects,img_path,overlap[gt_image_idxes.index(idx)],miss[:,gt_image_idxes.index(idx)])
            img = cv2.imread(img_path)
            vot.write(img)
        vot.release()
        np.savetxt(os.path.join(output_dir,'miss.txt'),miss,fmt="%d")
    
    else:
        dataset = lvx_object(root_path,split="testing")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        vot = cv2.VideoWriter(os.path.join(output_dir,"beicao_video.mp4"),fourcc,10,(3200,2400),True)
        for count,dt_anno in enumerate(dt_annos):
            idx = dt_anno['metadata']['token']
            print(idx)

            points = dataset.get_lidar(idx)
            points = points[points[:,1]<40,:]
            points = points[points[:,0]<40,:]
            points = points[points[:,1]>-40,:]
            points = points[points[:,0]>-40,:]
            points = points[points[:,2]>0.1,:]

            points_1 = dataset.get_lidar(f'{int(idx)-1}')
            points_1 = points_1[points_1[:,1]<40,:]
            points_1 = points_1[points_1[:,0]<40,:]
            points_1 = points_1[points_1[:,1]>-40,:]
            points_1 = points_1[points_1[:,0]>-40,:]
            points_1 = points_1[points_1[:,2]>0.1,:]

            if (count < len(dt_annos)-1):
                points_2 = dataset.get_lidar(f'{int(idx)+1}')
                points_2 = points_2[points_2[:,1]<40,:]
                points_2 = points_2[points_2[:,0]<40,:]
                points_2 = points_2[points_2[:,1]>-40,:]
                points_2 = points_2[points_2[:,0]>-40,:]
                points_2 = points_2[points_2[:,2]>0.1,:]
            else:
                points_2 = []
            dt_objects = dataset.get_label_objects(dt_anno)

            if not os.path.exists(os.path.join(output_dir,"beicao_imgs")):
                os.makedirs(os.path.join(output_dir,"beicao_imgs"))

            img_path = os.path.join(output_dir,f"beicao_imgs/lvx_{idx}.png")

            show_bev_objects(points,points_1,points_2,[],dt_objects,img_path,None, None)
            img = cv2.imread(img_path)
            vot.write(img)
        vot.release()
    
def lvx_vis_3d(gt_annos,dt_annos,output_dir):
    root_path = dt_annos[0]["metadata"]["image_prefix"]
    fig = mlab.figure(figure=None, bgcolor=(0,0,0), fgcolor=None, engine=None, size=(1600, 1000))

    if gt_annos is not None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        vot = cv2.VideoWriter(os.path.join(output_dir,"bev_video_3d.mp4"),fourcc,10,(1600,1000),True)
        dataset = lvx_object(root_path)
        gt_image_idxes = [str(info["token"]) for info in gt_annos]

        for i in range(len(gt_annos)):
            print(gt_annos[i]['token'])
            points = dataset.get_lidar(gt_annos[i]['token'])
            points = points[points[:,2]>0.1,:]
            gt_annos[i] = remove_dontcare(gt_annos[i])
            fig = draw_lidar(points,color=None,fig=fig, pts_mode = 'sphere')
            gt_corners = center_to_corner_box3d(gt_annos[i]['location'],gt_annos[i]['dimensions'], gt_annos[i]['rotation_y'])
            fig = draw_gt_boxes3d(gt_corners, gt_annos[i]['id'],color=(0,0,1),fig=fig)
            dt_corners = center_to_corner_box3d(dt_annos[i]['location'],dt_annos[i]['dimensions'], dt_annos[i]['rotation_y'])
            fig = draw_gt_boxes3d(dt_corners, dt_annos[i]['track_id'],color=(0,1,0),fig=fig)
            img_path = os.path.join(output_dir,f"bev_imgs/lvx_{gt_annos[i]['token']}_3d.png")
            mlab.savefig(img_path)
            mlab.clf(figure=fig)
            img = cv2.imread(img_path)
            vot.write(img)
        vot.release()
    else:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        vot = cv2.VideoWriter(os.path.join(output_dir,"beicao_video_3d.mp4"),fourcc,10,(1600,1000),True)
        dataset = lvx_object(root_path,split="testing")

        for i in range(len(dt_annos)):
            idx = dt_annos[i]['metadata']['token']
            print(idx)
            points = dataset.get_lidar(idx)
            plane = points[points[:,2]<0.1,:]
            points = points[points[:,2]>0.1,:]
            fig = draw_lidar(points,color=None,fig=fig, pts_mode = 'sphere')
            fig = draw_lidar(plane,color=None,fig=fig)
            dt_corners = center_to_corner_box3d(dt_annos[i]['location'],dt_annos[i]['dimensions'], dt_annos[i]['rotation_y'])
            color_list = np.zeros((dt_annos[i]['track_id'].shape[0],3))
            for j in range(dt_annos[i]['track_id'].shape[0]):
                np.random.seed(int(dt_annos[i]['track_id'][j]))
                color_list[j] = np.random.rand(3)

            view_setting = [180+30*np.sin(np.pi/100.*i),50+20*np.sin(np.pi/200.*i),60-20*np.sin(np.pi/100*i)]
            print(view_setting)
            fig = draw_gt_boxes3d(dt_corners, dt_annos[i]['track_id'],line_width=3,fig=fig,color_list=color_list,view_setting=view_setting)
            img_path = os.path.join(output_dir,f"beicao_imgs/lvx_{idx}_3d.png")
            mlab.savefig(img_path)
            mlab.clf(figure=fig)
            img = cv2.imread(img_path)
            vot.write(img)
        vot.release()

def remove_dontcare(image_anno):
    indice = []
    for i in range(len(image_anno['name'])):
        if image_anno['name'][i] == "DontCare":
            indice.append(i)
    
    image_anno['name'] = np.delete(image_anno['name'],indice)
    image_anno['location'] = np.delete(image_anno['location'],indice,axis=0)
    image_anno['dimensions'] = np.delete(image_anno['dimensions'],indice,axis=0)
    image_anno['rotation_y'] = np.delete(image_anno['rotation_y'],indice)

    if 'id' in image_anno.keys():
        image_anno['id'] = np.delete(image_anno['id'],indice)
    else:
        image_anno['track_id'] = np.delete(image_anno['track_id'],indice)

    return image_anno