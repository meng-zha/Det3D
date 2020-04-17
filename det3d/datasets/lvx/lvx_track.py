from det3d.datasets.utils.eval import box3d_overlap
from det3d.core.bbox.box_np_ops import center_to_corner_box3d
import numpy as np

def lvx_track(annos):
    """ 将上一帧的next预测与当前帧对比，将当前帧的prev预测与上一帧对比 """
    new_id = 0
    threshold = -0.75
    threshold_2 = -0.25

    # 上一帧没有匹配的检测框
    vir_annos = {'track_id':np.array([-2]).reshape(1),
                 'location':np.array([[9999,9999,-2]]).reshape(1,3),
                 'dimensions':np.array([[0,0,0]]).reshape(1,3),
                 'rotation_y':np.array([0]).reshape(1),
                 'location_1':np.array([[9999,9999,-2]]).reshape(1,3),
                 'dimensions_1':np.array([[0,0,0]]).reshape(1,3),
                 'rotation_y_1':np.array([0]).reshape(1),
                 'location_2':np.array([[9999,9999,-2]]).reshape(1,3),
                 'dimensions_2':np.array([[0,0,0]]).reshape(1,3),
                 'rotation_y_2':np.array([0]).reshape(1),
                 'age':np.array([-10000]).reshape(1) # 寿命暂定为6帧
                 }
    
    for i,anno in enumerate(annos):
        if i == 0:
            anno["track_id"] = np.arange(0,anno["name"].shape[0])
            new_id+=anno["name"].shape[0]
        else:
            boxes = anno_to_boxes(anno,'')
            boxes_1 = anno_to_boxes(annos[i-1],'_2')

            boxes_2 = anno_to_boxes(anno,'_1')
            boxes_3 = anno_to_boxes(annos[i-1],'')
            track_ids = -1*np.ones(boxes.shape[0])
            scores = -1*np.ones(boxes.shape[0])
            flag = []

            # 基准帧为过去一帧的当前检测
            past_predict = get_track_scores(boxes_2,boxes_3)
            get_track(past_predict,-0.125,track_ids,scores, annos[i-1],flag)
            
            predict_pre = get_track_scores(boxes,boxes_1)
            get_track(predict_pre,-0.125,track_ids,scores,annos[i-1],flag)

            # 前一帧没有被匹配的加入到vir中
            for j in range(len(annos[i-1]['track_id'])):
                if j not in flag:
                    if (vir_annos['track_id'] == annos[i-1]['track_id'][j]).any():
                        # 如果此id已经在之前的vir出现过，则更新
                        old = np.where(vir_annos['track_id']==annos[i-1]['track_id'][j])
                        vir_annos['location'][old] = annos[i-1]['location'][j]
                        vir_annos['dimensions'][old] = annos[i-1]['dimensions'][j]
                        vir_annos['rotation_y'][old] = annos[i-1]['rotation_y'][j]
                        vir_annos['location_1'][old] = annos[i-1]['location_1'][j]
                        vir_annos['dimensions_1'][old] = annos[i-1]['dimensions_1'][j]
                        vir_annos['rotation_y_1'][old] = annos[i-1]['rotation_y_1'][j]
                        vir_annos['location_2'][old] = annos[i-1]['location_2'][j]
                        vir_annos['dimensions_2'][old] = annos[i-1]['dimensions_2'][j]
                        vir_annos['rotation_y_2'][old] = annos[i-1]['rotation_y_2'][j]
                        vir_annos['age'][old] = 0 #寿命清0
                    else:
                        # 否则，则将其添加到vir_annos里
                        vir_annos['track_id']=np.concatenate([vir_annos['track_id'],[annos[i-1]['track_id'][j]]],axis=-1)
                        vir_annos['location']=np.concatenate([vir_annos['location'],[annos[i-1]['location'][j]]],axis=0)
                        vir_annos['dimensions']=np.concatenate([vir_annos['dimensions'],[annos[i-1]['dimensions'][j]]],axis=0)
                        vir_annos['rotation_y']=np.concatenate([vir_annos['rotation_y'],[annos[i-1]['rotation_y'][j]]],axis=-1)
                        vir_annos['location_1']=np.concatenate([vir_annos['location_1'],[annos[i-1]['location_1'][j]]],axis=0)
                        vir_annos['dimensions_1']=np.concatenate([vir_annos['dimensions_1'],[annos[i-1]['dimensions_1'][j]]],axis=0)
                        vir_annos['rotation_y_1']=np.concatenate([vir_annos['rotation_y_1'],[annos[i-1]['rotation_y_1'][j]]],axis=-1)
                        vir_annos['location_2']=np.concatenate([vir_annos['location_2'],[annos[i-1]['location_2'][j]]],axis=0)
                        vir_annos['dimensions_2']=np.concatenate([vir_annos['dimensions_2'],[annos[i-1]['dimensions_2'][j]]],axis=0)
                        vir_annos['rotation_y_2']=np.concatenate([vir_annos['rotation_y_2'],[annos[i-1]['rotation_y_2'][j]]],axis=-1)
                        vir_annos['age']=np.concatenate([vir_annos['age'],[0]],axis=-1)
            
            vir_boxes = anno_to_boxes(vir_annos,'')
            flag_vir = []
            # 与virtual框进行阈值更低的匹配
            # if anno['metadata']['token'] == '642':
            #     print(boxes_2,vir_boxes)
            #     import pdb; pdb.set_trace()
            virtual = get_track_scores(boxes_2,vir_boxes)
            get_track(virtual,-0.5,track_ids,scores,vir_annos,flag_vir)

            # 删除virtual中被成功匹配的框,被具现化了
            vir_annos['location'] = np.delete(vir_annos['location'],flag_vir,axis=0)
            vir_annos['dimensions'] = np.delete(vir_annos['dimensions'],flag_vir,axis=0)
            vir_annos['rotation_y'] = np.delete(vir_annos['rotation_y'],flag_vir,axis=0)
            vir_annos['location_1'] = np.delete(vir_annos['location_1'],flag_vir,axis=0)
            vir_annos['dimensions_1'] = np.delete(vir_annos['dimensions_1'],flag_vir,axis=0)
            vir_annos['rotation_y_1'] = np.delete(vir_annos['rotation_y_1'],flag_vir,axis=0)
            vir_annos['location_2'] = np.delete(vir_annos['location_2'],flag_vir,axis=0)
            vir_annos['dimensions_2'] = np.delete(vir_annos['dimensions_2'],flag_vir,axis=0)
            vir_annos['rotation_y_2'] = np.delete(vir_annos['rotation_y_2'],flag_vir,axis=0)
            vir_annos['track_id'] = np.delete(vir_annos['track_id'],flag_vir,axis=0)
            vir_annos['age'] = np.delete(vir_annos['age'],flag_vir,axis=0)

            # 向下一帧移动   
            vir_annos['location_1'][:,:2]= vir_annos['location'][:,:2].copy()
            vir_annos['location'][:,:2] = vir_annos['location_2'][:,:2].copy()
            flag_vir = []
            for j in range((len(vir_annos['track_id']))):
                vir_annos['location_2'][j,:2] = 2*vir_annos['location'][j,:2]-vir_annos['location_1'][j,:2]
                vir_annos['age'][j] += 1
                if vir_annos['location'][j,0]<-40 or vir_annos['location'][j,0]>40 or vir_annos['location'][j,1]<-40 or vir_annos['location'][j,1]>40:
                    # 如果出了边界，去掉这个人
                    flag_vir.append(j)
                if vir_annos['age'][j] > 6 and j not in flag_vir:
                    # age>6 去掉
                    flag_vir.append(j)
            
            vir_annos['location'] = np.delete(vir_annos['location'],flag_vir,axis=0)
            vir_annos['dimensions'] = np.delete(vir_annos['dimensions'],flag_vir,axis=0)
            vir_annos['rotation_y'] = np.delete(vir_annos['rotation_y'],flag_vir,axis=0)
            vir_annos['location_1'] = np.delete(vir_annos['location_1'],flag_vir,axis=0)
            vir_annos['dimensions_1'] = np.delete(vir_annos['dimensions_1'],flag_vir,axis=0)
            vir_annos['rotation_y_1'] = np.delete(vir_annos['rotation_y_1'],flag_vir,axis=0)
            vir_annos['location_2'] = np.delete(vir_annos['location_2'],flag_vir,axis=0)
            vir_annos['dimensions_2'] = np.delete(vir_annos['dimensions_2'],flag_vir,axis=0)
            vir_annos['rotation_y_2'] = np.delete(vir_annos['rotation_y_2'],flag_vir,axis=0)
            vir_annos['track_id'] = np.delete(vir_annos['track_id'],flag_vir,axis=0)
            vir_annos['age'] = np.delete(vir_annos['age'],flag_vir,axis=0)

            for index in range(track_ids.shape[0]):
                if track_ids[index] == -1:
                    print(anno['metadata']['token'],index,new_id,anno['location'][index])
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
            dist = ((corners[i]-corners_1[j])**2).sum()
            if dist > max_d:
                max_d = dist
    return max_d
    
def anno_to_boxes(anno,flag):
    boxes = np.concatenate([anno[f'location{flag}'],anno[f'dimensions{flag}'],anno[f'rotation_y{flag}'][...,np.newaxis]],axis=1)
    return boxes

def get_track(iou, threshold, track, scores, anno, flag):
    num1 = 0
    num2 = 0
    while(num1<iou.shape[0] and num2<iou.shape[1]):
        loc_max = iou.argmax()
        max_iou = [loc_max//iou.shape[1],loc_max%iou.shape[1]]
        score = iou[max_iou[0],max_iou[1]]
        if score <threshold:
            break
        # track_ids 需要没有被占用过,并且前一帧的flag没被用过
        if track[max_iou[0]] != -1:
            iou[max_iou[0],:] = -1
            num1+=1
        elif max_iou[1] in flag:
            iou[:,max_iou[1]] = -1
            num2+=1
        else:
            track[max_iou[0]] = anno['track_id'][max_iou[1]]
            scores[max_iou[0]] = score
            flag.append(max_iou[1])
            iou[max_iou[0],:] = -1
            iou[:,max_iou[1]] = -1
            num1+=1
            num2+=1