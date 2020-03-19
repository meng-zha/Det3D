from det3d.datasets.utils.eval import box3d_overlap
import numpy as np

def lvx_track(annos):
    """ 将上一帧的next预测与当前帧对比，将当前帧的prev预测与上一帧对比 """
    new_id = 0
    threshold = 0.1
    
    for i,anno in enumerate(annos):
        if i == 0:
            anno["track_id"] = np.arange(0,anno["name"].shape[0])
            new_id+=anno["name"].shape[0]
        else:
            # 基准帧为过去一帧的当前检测
            boxes = anno_to_boxes(anno,'')
            boxes_1 = anno_to_boxes(annos[i-1],'_2')
            predict_pre = box3d_overlap(boxes,boxes_1,z_axis=2,z_center=0.5)
            track_ids = -1*np.ones(boxes.shape[0])
            scores = np.zeros(boxes.shape[0])
            
            num = 0
            while(num<boxes.shape[0] and num<boxes_1.shape[0]):
                loc_max = predict_pre.argmax()
                max_iou = [loc_max//boxes_1.shape[0],loc_max%boxes_1.shape[0]]
                score = predict_pre[max_iou[0],max_iou[1]]
                if score <threshold:
                    break
                track_ids[max_iou[0]] = annos[i-1]['track_id'][max_iou[1]]
                scores[max_iou[0]] = score
                predict_pre[max_iou[0],:] = 0
                predict_pre[:,max_iou[1]] = 0
            
            for index in range(track_ids.shape[0]):
                if track_ids[index] == -1:
                    track_ids[index] = new_id
                    new_id += 1
            anno["track_id"] = track_ids
    
    return annos



    
def anno_to_boxes(anno,flag):
    boxes = np.concatenate([anno[f'location{flag}'],anno[f'dimensions{flag}'],anno[f'rotation_y{flag}'][...,np.newaxis]],axis=1)
    return boxes