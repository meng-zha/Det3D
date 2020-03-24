import math
import torch
import torch.nn as nn
from det3d.core import bbox_overlaps
from det3d.ops.nms.nms_gpu import rotate_iou_gpu

from ..registry import LOSSES
from .utils import weighted_loss


@weighted_loss
def iou_loss(pred, target, eps=1e-6):
    """IoU loss.
    Computing the IoU loss between a set of predicted bboxes and target bboxes.
    The loss is calculated as negative log of IoU.
    Args:
        pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (Tensor): Corresponding gt bboxes, shape (n, 4).
        eps (float): Eps to avoid log(0).
    Return:
        Tensor: Loss tensor.
    """
    ious = bbox_overlaps(pred, target, is_aligned=True).clamp(min=eps)
    loss = -ious.log()
    return loss


@weighted_loss
def bounded_iou_loss(pred, target, beta=0.2, eps=1e-3):
    """Improving Object Localization with Fitness NMS and Bounded IoU Loss,
    https://arxiv.org/abs/1711.00164.
    Args:
        pred (tensor): Predicted bboxes.
        target (tensor): Target bboxes.
        beta (float): beta parameter in smoothl1.
        eps (float): eps to avoid NaN.
    """
    pred_ctrx = (pred[:, 0] + pred[:, 2]) * 0.5
    pred_ctry = (pred[:, 1] + pred[:, 3]) * 0.5
    pred_w = pred[:, 2] - pred[:, 0] + 1
    pred_h = pred[:, 3] - pred[:, 1] + 1
    with torch.no_grad():
        target_ctrx = (target[:, 0] + target[:, 2]) * 0.5
        target_ctry = (target[:, 1] + target[:, 3]) * 0.5
        target_w = target[:, 2] - target[:, 0] + 1
        target_h = target[:, 3] - target[:, 1] + 1

    dx = target_ctrx - pred_ctrx
    dy = target_ctry - pred_ctry

    loss_dx = 1 - torch.max(
        (target_w - 2 * dx.abs()) / (target_w + 2 * dx.abs() + eps),
        torch.zeros_like(dx),
    )
    loss_dy = 1 - torch.max(
        (target_h - 2 * dy.abs()) / (target_h + 2 * dy.abs() + eps),
        torch.zeros_like(dy),
    )
    loss_dw = 1 - torch.min(target_w / (pred_w + eps), pred_w / (target_w + eps))
    loss_dh = 1 - torch.min(target_h / (pred_h + eps), pred_h / (target_h + eps))
    loss_comb = torch.stack([loss_dx, loss_dy, loss_dw, loss_dh], dim=-1).view(
        loss_dx.size(0), -1
    )

    loss = torch.where(
        loss_comb < beta, 0.5 * loss_comb * loss_comb / beta, loss_comb - 0.5 * beta
    )
    return loss


@LOSSES.register_module
class IoULoss(nn.Module):
    def __init__(self, eps=1e-6, reduction="mean", loss_weight=1.0):
        super(IoULoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(
        self,
        pred,
        target,
        weight=None,
        avg_factor=None,
        reduction_override=None,
        **kwargs
    ):
        if weight is not None and not torch.any(weight > 0):
            return (pred * weight).sum()  # 0
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction
        loss = self.loss_weight * iou_loss(
            pred,
            target,
            weight,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs
        )
        return loss

@LOSSES.register_module
class DIoULoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(DIoULoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(
        self,
        boxes,
        qboxes,
        weights,
        **kwargs
    ):
        assert boxes.size() == qboxes.size()
        batch_size = boxes.size()[0]
        boxes = boxes.view(-1,7)
        qboxes = qboxes.view(-1,7)
        weights = weights.view(-1)
        bev_boxes = [0,1,3,4,6]

        pred = torch.cat([boxes[:,0:1]-boxes[:,3:4]/2,boxes[:,1:2]-boxes[:,4:5]/2,boxes[:,0:1]+boxes[:,3:4]/2,boxes[:,1:2]+boxes[:,4:5]/2],axis=1)
        target = torch.cat([qboxes[:,0:1]-qboxes[:,3:4]/2,qboxes[:,1:2]-qboxes[:,4:5]/2,qboxes[:,0:1]+qboxes[:,3:4]/2,qboxes[:,1:2]+qboxes[:,4:5]/2],axis=1)
        z_axis = 2
        z_center = 0.5

        ious = bbox_overlaps(pred, target, is_aligned=True)
        ious = ious * torch.cos(boxes[:,6]-qboxes[:,6])

        x_axis = torch.cat([pred[:,[0,2]],target[:,[0,2]]],axis=1)
        x_min = x_axis.min(axis=1).values
        x_max = x_axis.max(axis=1).values
        y_axis = torch.cat([pred[:,[1,3]],target[:,[1,3]]],axis=1)
        y_min = y_axis.min(axis=1).values
        y_max = y_axis.max(axis=1).values

        dist_max = (x_max-x_min)**2 + (y_max-y_min)**2
        center = (boxes[:,0]-qboxes[:,0])**2+(boxes[:,1]-qboxes[:,1])**2

        loss = weights*(1-ious+(center+1e-10)/(dist_max+1e-10))

        # N = boxes.shape[0]
        # loss = torch.zeros(N)
        # for i in range(N):
        #     print (i)
        #     rinc = devRotateIoU(boxes[i, bev_boxes], qboxes[i, bev_boxes])
        #     if rinc > 0:
        #         min_z = min(
        #             boxes[i, z_axis] + boxes[i, z_axis + 3] * (1 - z_center),
        #             qboxes[i, z_axis] + qboxes[i, z_axis + 3] * (1 - z_center),
        #         )
        #         max_z = max(
        #             boxes[i, z_axis] - boxes[i, z_axis + 3] * z_center,
        #             qboxes[i, z_axis] - qboxes[i, z_axis + 3] * z_center,
        #         )
        #         iw = min_z - max_z
        #         if iw > 0:
        #             area1 = boxes[i, 3] * boxes[i, 4] * boxes[i, 5]
        #             area2 = qboxes[i, 3] * qboxes[i, 4] * qboxes[i, 5]
        #             inc = iw * rinc
        #             ua = area1 + area2 - inc
        #             rinc = inc / ua
        #         else:
        #             rinc = 0.0
        #     dist = (boxes[i,0]-qboxes[i,0])**2 + (boxes[i,1]-qboxes[i,1])**2+(boxes[i,2]-qboxes[i,2])**2
            
        #     corners1 = torch.zeros((8,))
        #     corners2 = torch.zeros((8,))

        #     rbbox_to_corners(corners1,boxes[i,bev_boxes])
        #     rbbox_to_corners(corners2,qboxes[i,bev_boxes])

        #     max_dist = torch.Tensor([0.])
        #     for m in range(4):
        #         for n in range(4):
        #             tmp_d = (corners1[2*m]-corners2[2*n])**2 + (corners1[2*m+1]-corners2[2*n+1])**2
        #             if max_dist < tmp_d:
        #                 max_dist = tmp_d
            
        #     max_z = max(
        #         boxes[i, z_axis] + boxes[i, z_axis + 3] * (1 - z_center),
        #         qboxes[i, z_axis] + qboxes[i, z_axis + 3] * (1 - z_center),
        #     )
        #     min_z = max(
        #         boxes[i, z_axis] - boxes[i, z_axis + 3] * z_center,
        #         qboxes[i, z_axis] - qboxes[i, z_axis + 3] * z_center,
        #     )
        #     max_dist += (max_z-min_z)**2

        #     loss[i] = 1-rinc+dist/max_dist

        loss = loss.view(batch_size,-1)

        return self.loss_weight*loss

def trangle_area(a, b, c):
    return ((a[0] - c[0]) * (b[1] - c[1]) - (a[1] - c[1]) * (b[0] - c[0])) / 2.0


def area(int_pts, num_of_inter):
    area_val = 0.0
    for i in range(num_of_inter - 2):
        area_val += abs(
            trangle_area(
                int_pts[:2],
                int_pts[2 * i + 2 : 2 * i + 4],
                int_pts[2 * i + 4 : 2 * i + 6],
            )
        )
    return area_val


def sort_vertex_in_convex_polygon(int_pts, num_of_inter):
    if num_of_inter > 0:
        center = torch.zeros((2,))
        center[:] = 0.0
        for i in range(num_of_inter):
            center[0] += int_pts[2 * i]
            center[1] += int_pts[2 * i + 1]
        center[0] /= num_of_inter
        center[1] /= num_of_inter
        v = torch.zeros((2,))
        vs = torch.zeros((16,))
        for i in range(num_of_inter):
            v[0] = int_pts[2 * i] - center[0]
            v[1] = int_pts[2 * i + 1] - center[1]
            d = math.sqrt(v[0] * v[0] + v[1] * v[1])
            v[0] = v[0] / d
            v[1] = v[1] / d
            if v[1] < 0:
                v[0] = -2 - v[0]
            vs[i] = v[0]
        j = 0
        temp = 0
        for i in range(1, num_of_inter):
            if vs[i - 1] > vs[i]:
                temp = vs[i]
                tx = int_pts[2 * i]
                ty = int_pts[2 * i + 1]
                j = i
                while j > 0 and vs[j - 1] > temp:
                    vs[j] = vs[j - 1]
                    int_pts[j * 2] = int_pts[j * 2 - 2]
                    int_pts[j * 2 + 1] = int_pts[j * 2 - 1]
                    j -= 1

                vs[j] = temp
                int_pts[j * 2] = tx
                int_pts[j * 2 + 1] = ty


def line_segment_intersection(pts1, pts2, i, j, temp_pts):
    A = torch.zeros((2,))
    B = torch.zeros((2,))
    C = torch.zeros((2,))
    D = torch.zeros((2,))


    A[0] = pts1[2 * i]
    A[1] = pts1[2 * i + 1]

    B[0] = pts1[2 * ((i + 1) % 4)]
    B[1] = pts1[2 * ((i + 1) % 4) + 1]

    C[0] = pts2[2 * j]
    C[1] = pts2[2 * j + 1]

    D[0] = pts2[2 * ((j + 1) % 4)]
    D[1] = pts2[2 * ((j + 1) % 4) + 1]
    BA0 = B[0] - A[0]
    BA1 = B[1] - A[1]
    DA0 = D[0] - A[0]
    CA0 = C[0] - A[0]
    DA1 = D[1] - A[1]
    CA1 = C[1] - A[1]
    acd = DA1 * CA0 > CA1 * DA0
    bcd = (D[1] - B[1]) * (C[0] - B[0]) > (C[1] - B[1]) * (D[0] - B[0])
    if acd != bcd:
        abc = CA1 * BA0 > BA1 * CA0
        abd = DA1 * BA0 > BA1 * DA0
        if abc != abd:
            DC0 = D[0] - C[0]
            DC1 = D[1] - C[1]
            ABBA = A[0] * B[1] - B[0] * A[1]
            CDDC = C[0] * D[1] - D[0] * C[1]
            DH = BA1 * DC0 - BA0 * DC1
            Dx = ABBA * DC0 - BA0 * CDDC
            Dy = ABBA * DC1 - BA1 * CDDC
            temp_pts[0] = Dx / DH
            temp_pts[1] = Dy / DH
            return True
    return False


def point_in_quadrilateral(pt_x, pt_y, corners):
    ab0 = corners[2] - corners[0]
    ab1 = corners[3] - corners[1]

    ad0 = corners[6] - corners[0]
    ad1 = corners[7] - corners[1]

    ap0 = pt_x - corners[0]
    ap1 = pt_y - corners[1]

    abab = ab0 * ab0 + ab1 * ab1
    abap = ab0 * ap0 + ab1 * ap1
    adad = ad0 * ad0 + ad1 * ad1
    adap = ad0 * ap0 + ad1 * ap1

    return abab >= abap and abap >= 0 and adad >= adap and adap >= 0


def quadrilateral_intersection(pts1, pts2, int_pts):
    num_of_inter = 0
    for i in range(4):
        if point_in_quadrilateral(pts1[2 * i], pts1[2 * i + 1], pts2):
            int_pts[num_of_inter * 2] = pts1[2 * i]
            int_pts[num_of_inter * 2 + 1] = pts1[2 * i + 1]
            num_of_inter += 1
        if point_in_quadrilateral(pts2[2 * i], pts2[2 * i + 1], pts1):
            int_pts[num_of_inter * 2] = pts2[2 * i]
            int_pts[num_of_inter * 2 + 1] = pts2[2 * i + 1]
            num_of_inter += 1
    temp_pts = torch.zeros((2,))
    for i in range(4):
        for j in range(4):
            has_pts = line_segment_intersection(pts1, pts2, i, j, temp_pts)
            if has_pts:
                int_pts[num_of_inter * 2] = temp_pts[0]
                int_pts[num_of_inter * 2 + 1] = temp_pts[1]
                num_of_inter += 1

    return num_of_inter


def rbbox_to_corners(corners, rbbox):
    # generate clockwise corners and rotate it clockwise
    angle = rbbox[4]
    a_cos = math.cos(angle)
    a_sin = math.sin(angle)
    center_x = rbbox[0]
    center_y = rbbox[1]
    x_d = rbbox[2]
    y_d = rbbox[3]
    corners_x = torch.zeros((4,))
    corners_y = torch.zeros((4,))
    corners_x[0] = -x_d / 2
    corners_x[1] = -x_d / 2
    corners_x[2] = x_d / 2
    corners_x[3] = x_d / 2
    corners_y[0] = -y_d / 2
    corners_y[1] = y_d / 2
    corners_y[2] = y_d / 2
    corners_y[3] = -y_d / 2
    for i in range(4):
        corners[2 * i] = a_cos * corners_x[i] + a_sin * corners_y[i] + center_x
        corners[2 * i + 1] = -a_sin * corners_x[i] + a_cos * corners_y[i] + center_y


def inter(rbbox1, rbbox2):
    corners1 = torch.zeros((8,))
    corners2 = torch.zeros((8,))
    intersection_corners = torch.zeros((16,))

    rbbox_to_corners(corners1, rbbox1)
    rbbox_to_corners(corners2, rbbox2)

    num_intersection = quadrilateral_intersection(
        corners1, corners2, intersection_corners
    )
    sort_vertex_in_convex_polygon(intersection_corners, num_intersection)
    # print(intersection_corners.reshape([-1, 2])[:num_intersection])

    return area(intersection_corners, num_intersection)


def devRotateIoU(rbox1, rbox2):
    area1 = rbox1[2] * rbox1[3]
    area2 = rbox2[2] * rbox2[3]
    area_inter = inter(rbox1, rbox2)
    return area_inter / (area1 + area2 - area_inter)