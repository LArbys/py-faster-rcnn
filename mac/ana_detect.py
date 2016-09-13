import numpy as np
import pandas as pd
import os,sys,re

#make top box probability
def make_df(file_):
    df_ = pd.read_csv(file_,header=None,delimiter=" ",names=['entry','class','prob','x1','y1','x2','y2'])
    return df_.ix[ df_.groupby("entry",sort=False).agg(lambda x: np.argmax(x['prob'])).prob.values ]

#load the annotation file
def load_annos(file_):
    df_ = pd.read_csv(file_,header=None,delimiter=" ",names=['entry','class','x1','y1','x2','y2'])
    return df_

#IOU function
def iou(bbox1,bbox2):

    ixmin = np.maximum(bbox1[0], bbox2[0])
    iymin = np.maximum(bbox1[1], bbox2[1])
    
    ixmax = np.minimum(bbox1[2], bbox2[2])
    iymax = np.minimum(bbox1[3], bbox2[3])
    
    iw = np.maximum(ixmax - ixmin, 0.)
    ih = np.maximum(iymax - iymin, 0.)
    inters = iw*ih
    uni = ((bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1]) +
           (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1]) 
           - inters)
    return inters/uni

#determine IOU
def get_ious(DF1,DF2):
    b=[None]*DF1.index.size
    iy=0
    for ix,row1 in DF1.iterrows():
        row2=DF2.query('entry=={}'.format(int(row1.entry)))
        row1=row1[3:].values
        row2=row2.values[0][2:]
        b[iy] = iou(row1,row2)
        iy+=1
    return np.array(b)


PIZERO_DETECTIONS="end_01_detections.txt"
PIZERO_ANNOS="end_01_annotations.txt"

PIZERO_DET_DF=make_df(PIZERO_DETECTIONS)
PIZERO_ANN_DF=load_annos(PIZERO_ANNOS)
PIZERO_DET_DF['iou']=get_ious(PIZERO_DET_DF,PIZERO_ANN_DF)



