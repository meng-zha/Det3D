from det3d.datasets.utils.eval import box3d_overlap
from det3d.core.bbox.box_np_ops import center_to_corner_box3d
import numpy as np

def lvx_track(annos):
    """ 将上一帧的next预测与当前帧对比，将当前帧的prev预测与上一帧对比 """
    new_id = 0
    reserve_id = []
    threshold = -0.125
    threshold_1 = -0.325
    threshold_2 = -0.5

    birth_min = 3
    death_min = 6

    # 上一帧没有匹配的检测框，进入death
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
    
    # 新生track的id池，连续3帧才算为正式的
    birth_annos = {'track_id':np.array([-2]).reshape(1),
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
            flag = []

            # 先判断是否多个人交汇成一个人
            past_predict = get_track_scores(boxes_2,boxes_3)
            predict_pre = get_track_scores(boxes,boxes_1)

            for age,new_track in zip(birth_annos['age'],birth_annos['track_id']):
                ind = np.where(annos[i-1]['track_id'] == new_track)
                # 假阳性的不进行分裂
                predict_pre[:,ind] = -1
            split = []
            det_match = (predict_pre>threshold_1).sum(1).reshape(-1)
            track_match = (predict_pre>threshold_1).sum(0).reshape(-1)
            for det in range(det_match.shape[0]):
                indices = np.where(predict_pre[det]>threshold_1)[0]
                if det_match[det]>1 and (track_match[indices]<=1).all():
                    # 当前的detect与多个track匹配，说明多个人汇合一起，需要将detect分割 
                    # TODO: virtual box应该也算进来

                    # 如果汇合处的detect和其中一个track匹配度非常高，而一些很低，则说明只是个别没有检测到
                    stay = []
                    if predict_pre[det].max()>0.6:
                        for ind in range(len(indices)):
                            if predict_pre[det,indices[ind]] < predict_pre[det].max():
                                stay.append(ind)
                    indices = np.delete(indices,stay)

                    split.append(det)
                    detect = boxes[det].copy()
                    tracks = boxes_1[indices].copy()
                    loc_err = detect-tracks.mean(0)
                    new_detect = tracks.copy()
                    new_detect[:,:3] = tracks[:,:3] + loc_err[:3]
                    new_detect[:,6] = tracks[:,6] + loc_err[6]
                    new_detect_1 = boxes_3[indices].copy()
                    new_detect_2 = new_detect.copy()
                    new_detect_2[:,:3] = new_detect[:,:3]+tracks[:,:3]-new_detect_1[:,:3]
                    boxes = np.concatenate([boxes,new_detect],axis=0)
                    boxes_2 = np.concatenate([boxes_2,new_detect_1],axis=0)
                    anno['location'] = np.concatenate([anno['location'],new_detect[:,:3]],axis=0)
                    anno['dimensions'] = np.concatenate([anno['dimensions'],new_detect[:,3:6]],axis=0)
                    anno['rotation_y'] = np.concatenate([anno['rotation_y'],new_detect[:,6]],axis=0)
                    anno['location_1'] = np.concatenate([anno['location_1'],new_detect_1[:,:3]],axis=0)
                    anno['dimensions_1'] = np.concatenate([anno['dimensions_1'],new_detect_1[:,3:6]],axis=0)
                    anno['rotation_y_1'] = np.concatenate([anno['rotation_y_1'],new_detect_1[:,6]],axis=0)
                    anno['location_2'] = np.concatenate([anno['location_2'],new_detect_2[:,:3]],axis=0)
                    anno['dimensions_2'] = np.concatenate([anno['dimensions_2'],new_detect_2[:,3:6]],axis=0)
                    anno['rotation_y_2'] = np.concatenate([anno['rotation_y_2'],new_detect_2[:,6]],axis=0)
                    anno['name'] = np.concatenate([anno['name'],annos[i-1]['name'][indices]],axis=0)
                    anno['alpha'] = np.concatenate([anno['alpha'],annos[i-1]['alpha'][indices]],axis=0)
                    anno['bbox'] = np.concatenate([anno['bbox'],annos[i-1]['bbox'][indices]],axis=0)
                    anno['score'] = np.concatenate([anno['score'],anno['score'][det:det+1].repeat(indices.shape[0],axis=0)],axis=0)

            boxes = np.delete(boxes,split,axis=0)
            boxes_2 = np.delete(boxes_2,split,axis=0)
            track_ids = -1*np.ones(boxes.shape[0])
            scores = -1*np.ones(boxes.shape[0])

            # 将detect分割后同时修改anno
            anno = del_dict(anno,split)

            assert annos[i]['name'].shape[0] == boxes.shape[0]

            # 基准帧为过去一帧的当前检测
            past_predict = get_track_scores(boxes_2,boxes_3)
            # get_track(past_predict,threshold,track_ids,scores, annos[i-1],flag)
            
            predict_pre = get_track_scores(boxes,boxes_1)

            # T,T-1的可信度更高，T+1的可信度更低
            predict_pre = 0.6*past_predict+0.4*predict_pre
            for age,new_track in zip(birth_annos['age'],birth_annos['track_id']):
                ind = np.where(annos[i-1]['track_id'] == new_track)
                # new track不确定是否为假阳性，所以权重调低
                predict_pre[:,ind] = (predict_pre[:,ind]-threshold)*(1+age)/(birth_min+1) + threshold
            get_track(predict_pre,threshold,track_ids,scores,annos[i-1],flag)

            # 前一帧没有被匹配的加入到vir中，并且是已经确定为非假阳性的，不在birth里面
            for j in range(len(annos[i-1]['track_id'])):
                if j not in flag and annos[i-1]['track_id'][j] not in birth_annos['track_id']:
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
            get_track(virtual,threshold_2,track_ids,scores,vir_annos,flag_vir)

            # 删除virtual中被成功匹配的框,被具现化了
            vir_annos = del_dict(vir_annos,flag_vir)

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
                if vir_annos['age'][j] > death_min and j not in flag_vir:
                    # age>6 去掉
                    flag_vir.append(j)
            
            vir_annos = del_dict(vir_annos,flag_vir)

            #处理新生id
            flag_birth = []
            for j,new_track in enumerate(birth_annos['track_id']):
                ind = np.where(annos[i-1]['track_id'][flag] == new_track)[0]
                if ind.size>0 and birth_annos['age'][j]<birth_min-1:
                    birth_annos['age'][j] += 1
                elif new_track>=0:
                    # 若达到成年寿命，或者中途夭折，去掉
                    flag_birth.append(j)
                    if birth_annos['age'][j] < birth_min-1:
                        reserve_id.append(new_track)
            birth_annos['age'] = np.delete(birth_annos['age'],flag_birth)
            birth_annos['track_id'] = np.delete(birth_annos['track_id'],flag_birth)

            for index in range(track_ids.shape[0]):
                if track_ids[index] == -1:
                    print(anno['metadata']['token'],index,new_id,anno['location'][index])
                    if len(reserve_id) == 0:
                        # 假阳性的id再次使用
                        track_ids[index] = new_id
                        new_id += 1
                    else:
                        track_ids[index] = reserve_id[-1]
                        del reserve_id[-1]
                    birth_annos['track_id'] = np.concatenate([birth_annos['track_id'],[track_ids[index]]],axis=0)
                    birth_annos['age'] = np.concatenate([birth_annos['age'],[0]],axis=0)
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

def del_dict(anno,flag):
    anno['location'] = np.delete(anno['location'],flag,axis=0)
    anno['dimensions'] = np.delete(anno['dimensions'],flag,axis=0)
    anno['rotation_y'] = np.delete(anno['rotation_y'],flag,axis=0)
    anno['location_1'] = np.delete(anno['location_1'],flag,axis=0)
    anno['dimensions_1'] = np.delete(anno['dimensions_1'],flag,axis=0)
    anno['rotation_y_1'] = np.delete(anno['rotation_y_1'],flag,axis=0)
    anno['location_2'] = np.delete(anno['location_2'],flag,axis=0)
    anno['dimensions_2'] = np.delete(anno['dimensions_2'],flag,axis=0)
    anno['rotation_y_2'] = np.delete(anno['rotation_y_2'],flag,axis=0)

    if 'name' in anno.keys():
        anno['name'] = np.delete(anno['name'],flag,axis=0)
        anno['alpha'] = np.delete(anno['alpha'],flag,axis=0)
        anno['score'] = np.delete(anno['score'],flag,axis=0)
        anno['bbox'] = np.delete(anno['bbox'],flag,axis=0)
    
    if 'age' in anno.keys():
        anno['track_id'] = np.delete(anno['track_id'],flag,axis=0)
        anno['age'] = np.delete(anno['age'],flag,axis=0)
    
    return anno