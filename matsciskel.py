#!/usr/bin/python2.6
import sys,os,cv,cv2
import numpy as np
sys.path.insert(0,os.getcwd() + '/gco');
import gco

def select_region(mat,reg):
    labels = mat.copy()
    labels[labels!=reg]=-1
    labels[labels==reg]=255
    labels[labels==-1]=0
    return labels.astype('uint8')

def data_term(seed,d):
    if d > 0:
        data = cv2.morphologyEx(select_region(seed,0), 
                                cv2.MORPH_DILATE, 
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(d,d)))
    else:
        data = select_region(seed,0)
    num_labels = int(seed.max()+1)
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
    # inimg = cv.LoadImageM("seq1/img/image0091.tif")
    inimg = cv.LoadImageM("seq3/img/image0041.png")
    im = cv.CreateMat(inimg.rows, inimg.cols, cv.CV_8U)
    cv.CvtColor(inimg,im, cv.CV_RGB2GRAY)

    seed=np.genfromtxt("seq3/labels/image0040.label",dtype='int16')
    # seed=np.genfromtxt("seq1/labels/image0090.label",dtype='int16')

    num_labels = seed.max()+1

    data = data_term(seed,10)

    # for i in range(0,num_labels) :
    #     print str(i)+": "+str(data[:,:,i].max())+" "+str(data[:,:,i].min())
    #     cv2.imwrite("test"+str(i)+".png",data[:,:,i])

    adj=np.eye(num_labels,dtype='int16')
    adj[0,:]=1
    adj[:,0]=1

    # adj=np.genfromtxt("../matscicut/adj90.label",dtype='int16')

    output = gco.graph_cut(data,np.asarray(im[:,:]),seed,adj,num_labels)

    # in1 = data
    # in2 = np.asarray(im[:,:])
    # in3 = seed
    # in4 = np.eye(num_labels,dtype='int16')
    
    # print in1.dtype
    # print in2.dtype
    # print in3.dtype
    # print in4.dtype

    # print in1.shape
    # print in2.shape
    # print in3.shape
    # print in4.shape

    # output = gco.graph_cut(in1,in2,in3,in4,7)


    return 0
 
if __name__ == '__main__':
    sys.exit(main(*sys.argv))

# im = np.asarray( im[:,:] )      # convert to numpy array
