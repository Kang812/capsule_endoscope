import os
os.chdir("/media/bgw2001/One Touch/project/HardCover_ObjectDetection/")

from ensemble_boxes import weighted_boxes_fusion
from glob import glob
from tqdm import tqdm

import pandas as pd
import numpy as np
import json

import warnings
#warnings.filterwarnings(action='ignore')
warnings.filterwarnings(action='default')

def extraction(df, file_name):
    try:
        tmp_df = df[df['file_name'] == file_name]
    
        with open("./data/test/" + file_name, "r") as f:
            data =  json.load(f)
        
        h = data['imageHeight']
        w = data['imageWidth']

        bbox = []
        label_list = []
        score_list = []

        scores = tmp_df['confidence'].tolist()
        class_ids = tmp_df['class_id'].tolist()
        xmins = tmp_df['point1_x'].tolist()
        ymins = tmp_df['point1_y'].tolist()
        xmaxs = tmp_df['point3_x'].tolist()
        ymaxs = tmp_df['point3_y'].tolist()

        for xmin, ymin, xmax, ymax, score, class_id in zip(xmins, ymins, xmaxs, ymaxs, scores, class_ids):
            bbox.append([round(xmin / w, 2), round(ymin / h,2), round(xmax / w, 2), round(ymax / h, 2)])
            label_list.append(class_id)
            score_list.append(score)

        return bbox, label_list, score_list

    except:
        pass

def emsanble(filename):

    #DetectoRS_e17_submission = pd.read_csv("./submission/DetectoRS_e17.csv")
    #RS17_bbox, RS17_label_list, RS17_score_list = extraction(DetectoRS_e17_submission, filename)

    DetectoRS_e18_submission = pd.read_csv("./submission/DetectoRS_e18.csv")
    RS18_bbox, RS18_label_list, RS18_score_list = extraction(DetectoRS_e18_submission, filename)
    
    #DetectoRS_e19_submission = pd.read_csv("./submission/DetectoRS_e19.csv")
    #RS19_bbox, RS19_label_list, RS19_score_list = extraction(DetectoRS_e19_submission, filename)

    faster_rcnn_submission = pd.read_csv("./submission/faster_rcnn_resnet50_submission_v1_2.csv")
    fr_bbox, fr_label_list, fr_score_list = extraction(faster_rcnn_submission, filename)

    centernet_submission = pd.read_csv("./submission/centernet_summisson.csv")
    cn_bbox, cn_label_list, cn_score_list = extraction(centernet_submission, filename)

    boxes_list = []
    #boxes_list.append(RS17_bbox)
    boxes_list.append(RS18_bbox)
    #boxes_list.append(RS19_bbox)
    boxes_list.append(fr_bbox)
    boxes_list.append(cn_bbox)

    scores_list  = []
    #scores_list.append(RS17_score_list)
    scores_list.append(RS18_score_list)
    #scores_list.append(RS19_score_list)
    scores_list.append(fr_score_list)
    scores_list.append(cn_score_list)

    labels_list  = []
    #labels_list.append(RS17_label_list)
    labels_list.append(RS18_label_list)
    #labels_list.append(RS19_label_list)
    labels_list.append(fr_label_list)
    labels_list.append(cn_label_list)

    weights = [40,40,40]

    iou_thr = 0.25
    skip_box_thr = 0.0001
    sigma = 0.1

    boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)

    return boxes, scores, labels

test = glob("./data/test/*.json")

file_name = []
class_id = []
confidence = []

point1_x = []
point1_y = []

point2_x = []
point2_y = []

point3_x = []
point3_y = []

point4_x = []
point4_y = []

for i in tqdm(range(len(test))):

    test_file = test[i]
    #test_file = test_file.replace("\\", "/")
    index = test_file.rfind("/")

    filename =  test_file[index + 1:]
    #print(filename)
    
    boxes, scores, labels = emsanble(filename)

    for box, score, label in zip(boxes, scores, labels):
        file_name.append(filename)
    
        with open("./data/test/" + filename, "r") as f:
            data =  json.load(f)
        
        h = data['imageHeight']
        w = data['imageWidth']
    

        point1_x.append(box[0] * w)
        point1_y.append(box[1] * h)
    
        point2_x.append(box[2] * w)
        point2_y.append(box[1] * h)

        point3_x.append(box[2] * w)
        point3_y.append(box[3] * h)

        point4_x.append(box[0] * w)
        point4_y.append(box[3] * h)

        class_id.append(int(label))
        confidence.append(score)

df = pd.DataFrame({"file_name": file_name,
                   "class_id" : class_id,
                   "confidence":confidence,
                   "point1_x" : point1_x,
                   "point1_y" : point1_y,
                   "point2_x" : point2_x,
                   "point2_y" : point2_y,
                   "point3_x" : point3_x,
                   "point3_y" : point3_y,
                   "point4_x" : point4_x,
                   "point4_y" : point4_y,


})

print(df.shape)
df.to_csv("/media/bgw2001/One Touch/project/HardCover_ObjectDetection/emsanble/emsanble_submission.csv", index = False)