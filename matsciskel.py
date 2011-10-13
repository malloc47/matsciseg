#!/usr/bin/python2.7
import sys,os,cv,cv2
import numpy as np
sys.path.insert(0,os.getcwd() + '/gco');
import gco

def select_region(mat,reg):
    labels = mat.copy()
    labels[labels!=reg]=0
    labels[labels==reg]=255
    return labels.astype('uint8')

def data_term(seed,d):
    data = select_region(seed,0)
    num_labels = int(seed.max())
    for l in range(1,num_labels) :
        label = select_region(seed,l)
        if d > 0 :
            se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(d,d))
            label = cv2.morphologyEx(label, cv2.MORPH_DILATE, se)
        data = np.dstack((data,label))
    return data

def display(im):
    cv2.namedWindow("tmp",cv2.CV_WINDOW_AUTOSIZE)
    cv2.imshow("tmp",im)
    cv2.waitKey(0)

def main(*args):
    inimg = cv.LoadImageM("seq3/img/image0041.png")
    im = cv.CreateMat(inimg.rows, inimg.cols, cv.CV_8U)
    cv.CvtColor(inimg,im, cv.CV_RGB2GRAY)

    seed=np.genfromtxt("seq3/labels/image0040.label",dtype='int16')

    data = data_term(seed,10)

    gco.graph_cut(data,np.asarray(im[:,:]),seed,np.eye(7,dtype='int16'),7)

    return 0
 
if __name__ == '__main__':
    sys.exit(main(*sys.argv))

# im = np.asarray( im[:,:] )      # convert to numpy array
