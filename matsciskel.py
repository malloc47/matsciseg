#!/usr/bin/python2.6
import sys,os,cv,cv2
import numpy as np
sys.path.insert(0,os.getcwd() + '/gco');
import gco

def select_region(mat,reg,maxval=255):
    labels = mat.copy()
    labels[labels!=reg]=-1
    labels[labels==reg]=maxval
    labels[labels==-1]=0
    return labels.astype('uint8')

def layer(unlayered):
    data = select_region(unlayered,0)
    num_labels = int(unlayered.max()+1)
    for l in range(1,num_labels) :
        label = select_region(unlayered,l)
        data = np.dstack((data,label))
    return data

# Modifies, in place, the layered image with op
# Can ignore the first layer (0) if desired
def layer_op(layered, op, dofirst=True):
    for i in  range(layered.shape[2]) if dofirst else range(1,layered.shape[2]):
        layered[:,:,i] = op(layered[:,:,i])


def display(im):
    cv2.namedWindow("tmp",cv2.CV_WINDOW_AUTOSIZE)
    cv2.imshow("tmp",im)
    cv2.waitKey(0)

def dilate(img,d):
    return cv2.morphologyEx(img, 
                            cv2.MORPH_DILATE, 
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(d,d)))

def adjacent(labels):
    adj = np.zeros((labels.max()+1,labels.max()+1),dtype='int16')
    def set_adj(i,j):
        adj[i,j] = 1
        adj[j,i] = 1

    # list of quad tuples that define the i,j ranges and the i,j transforms
    ranges = [ (range(0,labels.shape[0]-1),
                range(0,labels.shape[1]),
                lambda i:i+1, lambda j:j),

               (range(0,labels.shape[0]),
                range(0,labels.shape[1]-1),
                lambda i:i, lambda j:j+1),

               (range(1,labels.shape[0]),
                range(0,labels.shape[1]),
                lambda i:i-1, lambda j:j),

               (range(0,labels.shape[0]),
                range(1,labels.shape[1]),
                lambda i:i, lambda j:j-1),

               (range(0,labels.shape[0]-2), # don't know why there were included
                range(0,labels.shape[1]),
                lambda i:i+2, lambda j:j),

               (range(0,labels.shape[0]),
                range(0,labels.shape[1]-2),
                lambda i:i, lambda j:j+2) ]

    for rg in ranges:
        for i in rg[0]:
            for j in rg[1]:
                r = labels[i,j]
                l = labels[rg[2](i),rg[3](j)]
                if r != l : set_adj(r,l)

    return adj

def smoothFn(s1,s2,l1,l2,adj):
    if l1==l2 :
        return 0
    if not adj :
        return 10000000
        # print "P: "+str(s1)+" "+str(s2)+" "+str(l1)+" "+str(l2)
    # print "P: " + str(int(1.0/(max(float(s1),float(s2))+1.0) * 255.0))
    return int(1.0/(max(float(s1),float(s2))+1.0) * 255.0)
    # return int(1.0/float((abs(float(s1)-float(s2)) if abs(float(s1)-float(s2)) > 9 else 9)+1)*255.0)

def main(*args):
    d = 20
    # inimg = cv.LoadImageM("seq1/img/image0091.tif")
    inimg = cv.LoadImageM("seq1/img/stfl91alss1.tif")
    # inimg = cv.LoadImageM("seq3/img/image0041.png")
    im = cv.CreateMat(inimg.rows, inimg.cols, cv.CV_8U)
    cv.CvtColor(inimg,im, cv.CV_RGB2GRAY)

    # seed=np.genfromtxt("seq3/labels/image0040.label",dtype='int16')
    seed=np.genfromtxt("seq1/labels/image0090.label",dtype='int16')

    num_labels = seed.max()+1

    data = layer(seed)

    layer_op(data, lambda img : dilate(img,d) , dofirst=False)

    # for i in range(0,num_labels) :
    #     print str(i)+": "+str(data[:,:,i].max())+" "+str(data[:,:,i].min())
    #     cv2.imwrite("test"+str(i)+".png",data[:,:,i])

    adj = adjacent(seed)

    output = gco.graph_cut(data,np.asarray(im[:,:]),
                           seed,adj,num_labels)
    # np.savetxt("image0041.label",output,fmt='%1d')
    np.savetxt("image0091.label",output,fmt='%1d')

    return 0
 
if __name__ == '__main__':
    sys.exit(main(*sys.argv))

# im = np.asarray( im[:,:] )      # convert to numpy array
