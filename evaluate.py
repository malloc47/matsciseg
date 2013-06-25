#!/usr/bin/env python
import sys,os
sys.path.insert(0,os.path.join(os.getcwd(),'.'))
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from matsci.draw import label_to_bmp
from pymorph import thin
import math

def find_gt(path):
    file_name = os.path.basename(path)
    dir_names = os.path.dirname(path).split(os.sep)
    # walk backwards
    for l in [dir_names[0:i] for i in range(len(dir_names)+1)]:
        gt_name = os.path.join(*(l+['ground',file_name]))
        if os.path.exists(gt_name):
            return gt_name
    return None

def read_labels(path):
    labels = np.genfromtxt(path,dtype='int16')
    return (thin(label_to_bmp(labels) > 0) ,
            len(np.unique(labels)) )

def scores(gt,t,d):
    dt = distance_transform_edt(np.logical_not(gt))
    dt2 = distance_transform_edt(np.logical_not(t))
    tp = float((dt[t]<=d).sum())
    fp = float((dt[t]>d).sum())
    fn = float((dt2[gt]>d).sum())
    p = tp/(tp+fp)
    r = tp/(tp+fn)
    fm = 2*(p*r) / (p+r)
    if math.isnan(fm):
        fm = 0
    if math.isnan(p):
        p = 0
    if math.isnan(r):
        r = 0
    return (fm,p,r)

def main(*args):
    if(len(args) < 1):
        return 1

    for i in range(1,len(args)):
        path = args[i]
        gt_path = find_gt(path)
        score_path = os.path.splitext(path)[0]+'.score'
        edges,nedges = read_labels(path)
        gt,ngt = read_labels(gt_path)

        score_array = np.zeros((10,4),dtype='float64')

        size_diff = ngt - nedges
        for d in range(1,11):
            print(str(scores(gt,edges,d)+(size_diff,)))
            score_array[d-1,:] = scores(gt,edges,d)+(size_diff,)

        
        np.savetxt(score_path,score_array,fmt='%10.5f',delimiter=',')
    return 0

if __name__ == '__main__':
    sys.exit(main(*sys.argv))
