import json
import numpy as np
import argparse
from random import randint
import cv2

with open('../infos/directory.json') as fp: all_data_dir=json.load(fp)    


ANN_FILE_train = all_data_dir+'Annotations_hico/train_annotations.json' 
ANN_FILE_test=all_data_dir+'Annotations_hico/test_annotations.json'
with open(ANN_FILE_train) as fp:
    ANNOTATIONS_train = json.load(fp)
with open(ANN_FILE_test) as fp:
    ANNOTATIONS_test = json.load(fp)

    

    
OBJ_PATH_train_s = all_data_dir+'Object_Detections_hico/train/'
OBJ_PATH_test_s= all_data_dir+'Object_Detections_hico/test/' 


with open(all_data_dir+'hico_infos/hico_list_obj.json') as fp:
    list_obj = json.load(fp)
obj_list = dict([(value, int(key)-1) for key, value in list_obj.items()]) 
#所有物品的类别，编号。组成字典

VERB2ID={u'adjust': 0,
 u'assemble': 1,
 u'block': 2,
 u'blow': 3,
 u'board': 4,
 u'break': 5,
 u'brush_with': 6,
 u'buy': 7,
 u'carry': 8,
 u'catch': 9,
 u'chase': 10,
 u'check': 11,
 u'clean': 12,
 u'control': 13,
 u'cook': 14,
 u'cut': 15,
 u'cut_with': 16,
 u'direct': 17,
 u'drag': 18,
 u'dribble': 19,
 u'drink_with': 20,
 u'drive': 21,
 u'dry': 22,
 u'eat': 23,
 u'eat_at': 24,
 u'exit': 25,
 u'feed': 26,
 u'fill': 27,
 u'flip': 28,
 u'flush': 29,
 u'fly': 30,
 u'greet': 31,
 u'grind': 32,
 u'groom': 33,
 u'herd': 34,
 u'hit': 35,
 u'hold': 36,
 u'hop_on': 37,
 u'hose': 38,
 u'hug': 39,
 u'hunt': 40,
 u'inspect': 41,
 u'install': 42,
 u'jump': 43,
 u'kick': 44,
 u'kiss': 45,
 u'lasso': 46,
 u'launch': 47,
 u'lick': 48,
 u'lie_on': 49,
 u'lift': 50,
 u'light': 51,
 u'load': 52,
 u'lose': 53,
 u'make': 54,
 u'milk': 55,
 u'move': 56,
 u'no_interaction': 57,
 u'open': 58,
 u'operate': 59,
 u'pack': 60,
 u'paint': 61,
 u'park': 62,
 u'pay': 63,
 u'peel': 64,
 u'pet': 65,
 u'pick': 66,
 u'pick_up': 67,
 u'point': 68,
 u'pour': 69,
 u'pull': 70,
 u'push': 71,
 u'race': 72,
 u'read': 73,
 u'release': 74,
 u'repair': 75,
 u'ride': 76,
 u'row': 77,
 u'run': 78,
 u'sail': 79,
 u'scratch': 80,
 u'serve': 81,
 u'set': 82,
 u'shear': 83,
 u'sign': 84,
 u'sip': 85,
 u'sit_at': 86,
 u'sit_on': 87,
 u'slide': 88,
 u'smell': 89,
 u'spin': 90,
 u'squeeze': 91,
 u'stab': 92,
 u'stand_on': 93,
 u'stand_under': 94,
 u'stick': 95,
 u'stir': 96,
 u'stop_at': 97,
 u'straddle': 98,
 u'swing': 99,
 u'tag': 100,
 u'talk_on': 101,
 u'teach': 102,
 u'text_on': 103,
 u'throw': 104,
 u'tie': 105,
 u'toast': 106,
 u'train': 107,
 u'turn': 108,
 u'type_on': 109,
 u'walk': 110,
 u'wash': 111,
 u'watch': 112,
 u'wave': 113,
 u'wear': 114,
 u'wield': 115,
 u'zip': 116}
MATCHING_IOU = .5
NO_VERBS = 117




def get_detections(segment_key,flag):
    if flag=='train':
	key_ann='%.8i'%(segment_key)
	annotation = ANNOTATIONS_train[key_ann]
	cur_obj_path_s = OBJ_PATH_train_s + "HICO_train2015_%.8i.json" % (segment_key)
	SCORE_TH = 0.6
	SCORE_OBJ=0.3

	select_threshold=15
    elif flag=='test':
	key_ann='%.8i'%(segment_key)#格式化输出吧
	annotation = ANNOTATIONS_test[key_ann]
	cur_obj_path_s = OBJ_PATH_test_s + "HICO_test2015_%.8i.json" % (segment_key)
	SCORE_TH = 0.6
	SCORE_OBJ=0.3
	select_threshold=15

    annotation = clean_up_annotation(annotation)
    with open(cur_obj_path_s) as fp:detections = json.load(fp)
    #detections=[]列表表示，得到当前图片标注的检测结果
    
    img_H = detections['H']
    img_W = detections['W']
    shape=[img_W,img_H] 
    persons_d, objects_d = analyze_detections(detections,SCORE_TH,SCORE_OBJ)#分析检测到的对象分类得分是够大于预支
    d_p_boxes,scores_persons,class_id_humans = get_boxes_det(persons_d, img_H, img_W)
    d_o_boxes,scores_objects,class_id_objects = get_boxes_det(objects_d, img_H, img_W)#对图中所有的人和物进行编号
    
    try:
        d_o_boxes=np.concatenate([d_o_boxes,d_p_boxes]).tolist()
    except: 
        import pdb;pdb.set_trace()
    class_id_objects=np.concatenate([class_id_objects,class_id_humans]).tolist()
    scores_objects=np.concatenate([scores_objects,scores_persons]).tolist()
    
    if len(d_p_boxes)>select_threshold:
        d_p_boxes,scores_persons,class_id_humans= d_p_boxes[0:select_threshold],scores_persons[0:select_threshold],class_id_humans[0:select_threshold]
    if len(d_o_boxes)>select_threshold:
        d_o_boxes,scores_objects,class_id_objects= d_o_boxes[0:select_threshold],scores_objects[0:select_threshold],class_id_objects[0:select_threshold]
   #最多保留15个人和物的框
    
    #scores_objects.insert(0,1)
    return d_p_boxes,d_o_boxes,scores_persons,scores_objects,class_id_humans,class_id_objects,annotation,shape



    
    
def get_compact_detections(segment_key,flag):    
    d_p_boxes,d_o_boxes,scores_persons,scores_objects,class_id_humans,class_id_objects,annotation,shape=get_detections(segment_key,flag)
    img_W=shape[0]
    img_H=shape[1]
    no_person_dets = len(d_p_boxes)
    no_object_dets = len(d_o_boxes)
    persons_np = np.zeros([no_person_dets, 4], np.float32)
    objects_np = np.zeros([no_object_dets, 4], np.float32)
    if no_person_dets != 0:
	persons_np = np.array(d_p_boxes, np.float32)
 
		
    objects_np = np.array(d_o_boxes, np.float32) 
    persons_np=persons_np/ np.array([img_W, img_H, img_W, img_H])
    objects_np=objects_np/ np.array([img_W, img_H, img_W, img_H])
    return { 'person_bbx':persons_np, 'objects_bbx':objects_np,
	    'person_bbx_score':scores_persons,'objects_bbx_score':scores_objects,
            'class_id_objects':class_id_objects}
    
    
    
    
def get_attention_maps(segment_key,flag): 
    compact_detections=get_compact_detections(segment_key,flag)
    persons_np,objects_np=compact_detections['person_bbx'],compact_detections['objects_bbx']
    union_box=[]
    no_person_dets = len(persons_np)
    no_object_dets = len(objects_np)
    for dd_i in range(no_person_dets):
        for do_i in range(len(objects_np)):
            union_box.append(union_BOX(persons_np[dd_i],objects_np[do_i],segment_key))
    return np.concatenate(union_box)


    
    
    
def get_compact_label(segment_key,flag):    
  
    d_p_boxes,d_o_boxes,scores_persons,scores_objects,class_id_humans,class_id_objects,annotation,shape=get_detections(segment_key,flag)
    
    no_person_dets = len(d_p_boxes)#检测到人的个数 
    no_object_dets = len(d_o_boxes)#检测到物的个数
    labels_np = np.zeros([no_person_dets, no_object_dets, NO_VERBS], np.int32) #构建空矩阵形成人，物，动词的的三元组
    
    a_p_boxes = [ann['person_box'] for ann in annotation]
    iou_mtx = get_iou_mtx(a_p_boxes, d_p_boxes)#计算人的检测框与与标注框的IoU

    if no_person_dets != 0 and len(a_p_boxes)!=0:
        max_iou_for_each_det = np.max(iou_mtx, axis=0)#取每一个检测结果与真值IoU的最大值
        index_for_each_det  = np.argmax(iou_mtx, axis=0)#IoU最大值的编号
        for dd in range(no_person_dets):
            cur_max_iou = max_iou_for_each_det[dd]#判断与真值的IoU最大的检测框是否大于阈值，仅处理大于设置的MATCHING_IOU的检测框
            if cur_max_iou < MATCHING_IOU: 
                continue
            matched_ann = annotation[index_for_each_det[dd]]#找到每个人的检测框对应的注释
            hoi_anns = matched_ann['hois']
            # verbs with actions######
            object_hois = [oi for oi in hoi_anns if len(oi['obj_box']) != 0]#判断注释中是否有交互行为
             
            a_o_boxes = [oi['obj_box'] for oi in object_hois]#得到对应的物的检测框
            iou_mtx_o = get_iou_mtx(a_o_boxes, d_o_boxes)#取与真值的IoU最大的物体检测框


            if a_o_boxes and d_o_boxes:
                for do in range(len(d_o_boxes)):
                  for ao in range(len(a_o_boxes)):
                    cur_iou = iou_mtx_o[ao,do]
                    # enough iou
                    if cur_iou < MATCHING_IOU:
                        continue
                    current_hoi = object_hois[ao]#如果物体存在，且物体的IoU足够大，则认为存在这个人物交互行为
                    verb_idx = VERB2ID[current_hoi['verb']] #确定动词
                    labels_np[dd, do, verb_idx] = 1 #确定三元组标签成立
    

        comp_labels=labels_np.reshape(no_person_dets*(no_object_dets),NO_VERBS)
        labels_single=np.array([1 if i.any()==True else 0 for i in comp_labels])#根据comp_labels，得到人与物的单个标签
        labels_single=labels_single.reshape(np.shape(labels_single)[0],1)#类别与标签的矩阵变换
        return{'labels_all':labels_np,'labels_single':labels_single}
    else:#如果没有人，都是0了吧
        comp_labels=labels_np.reshape(no_person_dets*(no_object_dets+1),NO_VERBS)
        labels_single=np.array([1 if i.any()==True else 0 for i in comp_labels])
        labels_single=labels_single.reshape(np.shape(labels_single)[0],1)
        return{'labels_all':labels_np,'labels_single':labels_single}
    
def get_bad_detections(segment_key,flag): #Detections Without any person#    
    
    labels_all=get_compact_label(segment_key,flag)['labels_all']
    if labels_all.size==0:
        return True
    else:
        return False

        






def union_BOX(roi_pers,roi_objs,segment_key,H=64,W=64):
        assert H==W
        roi_pers=np.array(roi_pers*H,dtype=int)
        roi_objs=np.array(roi_objs*H,dtype=int)
        sample_box=np.zeros([1,2,H,W])
        sample_box[0,0,roi_pers[1]:roi_pers[3]+1,roi_pers[0]:roi_pers[2]+1]=100
        sample_box[0,1,roi_objs[1]:roi_objs[3]+1,roi_objs[0]:roi_objs[2]+1]=100
	return sample_box            

	
	


def clean_up_annotation(annotation):
    persons_dict = {}
    person_list=[]
    object_list=[]
    for hoi in annotation:
	
    	 
        box = hoi['person_bbx']
        box = [int(coord) for coord in box]
	box=clean_person(person_list,np.asarray(box))
	#import pdb;pdb.set_trace()
        dkey = tuple(box)
        objects = hoi['object']
        if len(objects['obj_bbx']) == 0: # no obj case
            cur_oi = {  'verb': hoi['Verbs'], 
                        'obj_box':[],
                        'obj_str': '',
			'obj_id':''
                        }
        else:
           cur_obj=clean_object(object_list,np.asarray(hoi['object']['obj_bbx'])) 
	   cur_oi = {  'verb': hoi['Verbs'], 
                        #'obj_box':[int(coord) for coord in hoi['object']['obj_bbx']],
                        'obj_box':cur_obj,
                        'obj_str': hoi['object']['obj_name'],
			'obj_id':int(obj_list[hoi['object']['obj_name']])
                        
			}
            
        if dkey in persons_dict:
            persons_dict[dkey]['hois'].append(cur_oi)
        else:
            persons_dict[dkey] = {'person_box': box, 'hois': [cur_oi]}
    

    pers_list = []
    for dkey in persons_dict:
        pers_list.append(persons_dict[dkey])
    #import pdb;pdb.set_trace()
    return pers_list
def clean_person(person_list,box):
	if len(person_list)==0:
		person_list.append(box)
		return box.tolist()
		
		
	else:
		for person in person_list:
			if len(box)==0 :
				if len(person)==0:
					return person.tolist()
			elif len(person)==0:
				continue
					
				
			else:
				IOU=IoU_box(person, box)
				if IOU>=0.5:
					return person.tolist()
		person_list.append(box)
		return box.tolist()
def clean_object(object_list,box):
	if len(object_list)==0:
		object_list.append(box)
		return box.tolist()
	else:
		for object in object_list:
			
			if len(box)==0 :
				if len(object)==0:
					return object.tolist()
			elif len(object)==0:
				continue
			else:
				IOU=IoU_box(object, box)
				if IOU>=0.5:
					return object.tolist()
		
		object_list.append(box)
		return box.tolist()

def get_boxes_det(dets, img_H, img_W): 
    boxes = []
    scores=[]
    class_no=[]
    for det in dets:
        top,left,bottom,right = det['box_coords']（左上，右下坐标）
	scores.append(det['score'])
	if len(det['class_str'].split())==2:
	        #import pdb;pdb.set_trace()
		str=det['class_str'].split()[0]+'_'+det['class_str'].split()[1]
	else:
		str=det['class_str']
	#得到物体类别名称



	class_no.append(int(obj_list[str]))#得到类别编号列表
        #left, top, right, bottom = left*img_W, top*img_H, right*img_W, bottom*img_H
        left, top, right, bottom = left, top, right, bottom
        boxes.append([left,top,right,bottom])
    #import pdb;pdb.set_trace()
    return boxes,scores,class_no
        

def get_iou_mtx(anns, dets):
    no_gt = len(anns)
    no_dt = len(dets)
    iou_mtx = np.zeros([no_gt, no_dt])

    for gg in range(no_gt):
        gt_box = anns[gg]
        for dd in range(no_dt):
            dt_box = dets[dd]
            iou_mtx[gg,dd] = IoU_box(gt_box,dt_box)

    return iou_mtx



def analyze_detections(detections,SCORE_TH,SCORE_OBJ):
	#分析检测的分类得分是否大于阈值
    persons = []
    objects = []
    
    for det in detections['detections']:
        if det['class_str'] == 'person':
            if det['score'] < SCORE_TH:
            	continue
            persons.append(det)
	
        else:
            if det['score'] < SCORE_OBJ:
            	continue
            objects.append(det)

    return persons, objects


def IoU_box(box1, box2):
    '''
    left1, top1, right1, bottom1 = box1
    left2, top2, right2, bottom2 = box2
 
    returns intersection over union
    '''
    try:
    	left1, top1, right1, bottom1 = box1
    	left2, top2, right2, bottom2 = box2
    except:
	IoU=0
	return IoU
 
    left_int = max(left1, left2)
    top_int = max(top1, top2)
 
    right_int = min(right1, right2)
    bottom_int = min(bottom1, bottom2)
 
    areaIntersection = max(0, right_int - left_int) * max(0, bottom_int - top_int)
 
    area1 = (right1 - left1) * (bottom1 - top1)
    area2 = (right2 - left2) * (bottom2 - top2)
     
    IoU = areaIntersection / float(area1 + area2 - areaIntersection)
    return IoU

def dry_run():
	print("Doing a test run to detect bad detections\n")
	
        with open(all_data_dir+'bad_detections_hico/bad_detections_train.json') as fp:
            bad_detections_train = json.load(fp)
        
        print("In training set object detector failed to detect any person in {} images".format(len(bad_detections_train)))
	
        
        with open(all_data_dir+'bad_detections_hico/bad_detections_test.json') as fp:
            bad_detections_test = json.load(fp)
        print("In testing set object detector failed to detect any person in {} images".format(len(bad_detections_test)))
        #import pdb;pdb.set_trace() 
        return bad_detections_train,bad_detections_test
        #读取图片数据中没有检测到人的图片



if __name__ == "__main__":
    new_anns = {}
    compact_dets={}
    att_maps={}
    parser=argparse.ArgumentParser()
    parser.add_argument('-t','--type_of_data',type=str,required=False,default='train',help="type_of_data")
    args=parser.parse_args()
    flag=args.type_of_data
    bad_detections_train,bad_detections_test=dry_run()
    b_d_tr,b_d_test=[int(l) for l in bad_detections_train],[int(l) for l in bad_detections_test]#记录没有检测到人的图片编号
    phases=['train','test']
    from tqdm import tqdm
    #import pdb;pdb.set_trace()
    for flag in phases:
	    if flag=='train':
		ALL_SEGS =ANNOTATIONS_train.keys()


	    elif flag=='test':
		ALL_SEGS =ANNOTATIONS_test.keys()
	    ALL_SEGS = [int(v) for v in ALL_SEGS]
	    ALL_SEGS.sort()
	    for segkey in tqdm(ALL_SEGS):
                #import pdb;pdb.set_trace() #设置进度条
	        if segkey in (b_d_tr+b_d_test):	#如果图片属于没有检测到人的图片
		    new_anns[segkey] = get_compact_label(segkey,flag) #
		    compact_dets[segkey] = get_compact_detections(segkey,flag)
		    att_maps[segkey] = get_attention_maps(segkey,flag)
                    #import pdb;pdb.set_trace()
		
    pass
