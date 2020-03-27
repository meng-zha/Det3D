from det3d.datasets.utils.eval import box3d_overlap
from det3d.core.bbox.box_np_ops import center_to_corner_box3d
import numpy as np

def lvx_track(annos):
    """ 将上一帧的next预测与当前帧对比，将当前帧的prev预测与上一帧对比 """
    new_id = 0
    threshold = -0.75
    threshold_2 = -0.25
    
    for i,anno in enumerate(annos):
        if i == 0:
            anno["track_id"] = np.arange(0,anno["name"].shape[0])
            new_id+=anno["name"].shape[0]
        else:
            # 基准帧为过去一帧的当前检测
            boxes = anno_to_boxes(anno,'')
            boxes_1 = anno_to_boxes(annos[i-1],'_2')
            predict_pre = get_track_scores(boxes,boxes_1)
            past_predict = get_track_scores(anno_to_boxes(anno,'_1'),anno_to_boxes(annos[i-1],''))

            predict_pre = (predict_pre+past_predict)/2
            track_ids = -1*np.ones(boxes.shape[0])
            scores = -1*np.ones(boxes.shape[0])
            
            num = 0
            while(num<boxes.shape[0] and num<boxes_1.shape[0]):
                loc_max = predict_pre.argmax()
                max_iou = [loc_max//boxes_1.shape[0],loc_max%boxes_1.shape[0]]
                score = predict_pre[max_iou[0],max_iou[1]]
                if score <threshold:
                    break
                track_ids[max_iou[0]] = annos[i-1]['track_id'][max_iou[1]]
                scores[max_iou[0]] = score
                predict_pre[max_iou[0],:] = -1
                predict_pre[:,max_iou[1]] = -1
                num+=1
            
            num = 0
            past_tom = get_track_scores(anno_to_boxes(anno,'_1'),anno_to_boxes(annos[i-2],'_2'))
            while(num<past_tom.shape[0] and num<past_tom.shape[1]):
                loc_max = past_tom.argmax()
                max_iou = [loc_max//past_tom.shape[1],loc_max%past_tom.shape[1]]
                score = past_tom[max_iou[0],max_iou[1]]
                if score <  threshold_2:
                    break
                if track_ids[max_iou[0]] == -1:
                    # track_ids 需要没有被占用过
                    if (track_ids == annos[i-2]['track_id'][max_iou[1]]).sum() == 0:
                        track_ids[max_iou[0]] = annos[i-2]['track_id'][max_iou[1]]
                        scores[max_iou[0]] = score
                past_tom[max_iou[0],:] = -1
                past_tom[:,max_iou[1]] = -1
                num+=1
                            
            for index in range(track_ids.shape[0]):
                if track_ids[index] == -1:
                    track_ids[index] = new_id
                    new_id += 1
            anno["track_id"] = track_ids
    
    return annos

def get_track_scores(boxes,boxes_1):
    ''' DIoU'''
    predict_pre = box3d_overlap(boxes,boxes_1,z_axis=2,z_center=0.5)

    corners = center_to_corner_box3d(boxes[:,:3],boxes[:,3:6],boxes[:,6])
    corners_1 = center_to_corner_box3d(boxes_1[:,:3],boxes_1[:,3:6],boxes_1[:,6])
    for i in range(boxes.shape[0]):
        for j in range(boxes_1.shape[0]):
            c = ((boxes[i,:3]-boxes_1[j,:3])**2).sum()
            d = get_farest(corners[i],corners_1[j])
            predict_pre[i,j] -= c/d
    return predict_pre


def get_farest(corners,corners_1):
    max_d = 0
    for i in range(8):
        for j in range(8):
            dist = ((corners[i]-corners_1[i])**2).sum()
            if dist > max_d:
                max_d = dist
    return max_d
    
def anno_to_boxes(anno,flag):
    boxes = np.concatenate([anno[f'location{flag}'],anno[f'dimensions{flag}'],anno[f'rotation_y{flag}'][...,np.newaxis]],axis=1)
    return boxes