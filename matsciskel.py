#!/usr/bin/python2.7
import sys,os,cv,cv2
import numpy as np
sys.path.insert(0,os.getcwd() + '/gco');
import gco
# from scipy import ndimage
import scipy

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
def skel(img):
    hits = [ np.array([[0, 0, 0], [0, 1, 0], [1, 1, 1]]), 
             np.array([[0, 0, 0], [1, 1, 0], [0, 1, 0]]) ]
    misses = [ np.array([[1, 1, 1], [0, 0, 0], [0, 0, 0]]),
               np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]]) ]

    for i in range(6):
        hits.append(np.transpose(hits[-2])[::-1, ...])
        misses.append(np.transpose(misses[-2])[::-1, ...])

    filters = zip(hits,misses)

    while True:
        prev = img
        for hit, miss in filters:
            filtered = ndimage.binary_hit_or_miss(img, hit, miss)
            img = np.logical_and(img, np.logical_not(filtered))
        if np.abs(prev-img).max() == 0:
            break
    
    return img

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

def label_to_bmp(labels):
    grad = np.gradient(labels)
    seg = np.maximum(abs(grad[0]),abs(grad[1]))
    return seg

def draw_on_img(img,bmp):
    out = img.copy()
    out[np.nonzero(bmp>0)] = (0,0,255)
    return out

def label_max(labels):
    sizes = []
    # check only indices above 0
    for l in range(1,labels.max()+1) :
        sizes.append((labels == l).sum())
    # increment index, as we ignored zero before
    return sizes.index(max(sizes))+1

def label_sizes(labels):
    sizes = []
    for l in range(0,labels.max()+1) :
        sizes.append((labels == l).sum())
    return sizes

def label_empty(labels):
    return [i for i,x in enumerate(label_sizes(labels)) if x == 0]

def region_transform(labels):
    unshifted = range(0,labels.max()+1)
    shifted = range(0,labels.max()+1)
    counter = 0
    empty = label_empty(labels)
    for l in unshifted :
        if l in empty :
            shifted[l] = -1
        else :
            shifted[l] = counter
            counter = counter + 1
    return zip(unshifted,shifted)

def region_shift(regions,transform) :
    labels = np.zeros(regions.shape,dtype=regions.dtype)
    for t in transform : 
        if (t[0] < 0 or t[1] < 0) : continue
        labels[regions == t[0]] = t[1]
    return labels

def region_clean(regions):
    # out = np.ones(regions.shape,dtype=regions.dtype) * -1
    out = np.zeros(regions.shape,dtype=regions.dtype)
    for l in range(regions.max()+1) :
        layer = (regions==l)
        if layer.any() :
            labeled = ndimage.label(layer>0)[0]
            # copy only the largest connected component
            out[np.nonzero(labeled == label_max(labeled))] = l
    return out

def smoothFn(s1,s2,l1,l2,adj):
    if l1==l2 :
        return 0
    if not adj :
        return 10000000
    return int(1.0/(max(float(s1),float(s2))+1.0) * 255.0)
    # return int(1.0/float((abs(float(s1)-float(s2)) if abs(float(s1)-float(s2)) > 9 else 9)+1)*255.0)

def main(*args):
    d = 10
    im = cv.LoadImageM("seq1/img/image0091.tif")
    # im = cv.LoadImageM("seq1/img/stfl91alss1.tif")
    # inimg = cv.LoadImageM("seq3/img/image0041.png")
    im_gray = cv.CreateMat(im.rows, im.cols, cv.CV_8U)
    cv.CvtColor(im,im_gray, cv.CV_RGB2GRAY)
    im_gray = np.asarray(im_gray[:,:])
    im = np.asarray(im[:,:])

    # seed=np.genfromtxt("seq3/labels/image0040.label",dtype='int16')
    seed = np.genfromtxt("seq1/labels/image0090.label",dtype='int16')
    seed = region_clean( region_shift(seed,region_transform(seed)) )

    num_labels = seed.max()+1

    data = layer(seed)

    layer_op(data, lambda img : dilate(img,d) , dofirst=False)

    # for i in range(0,num_labels) :
    #     print str(i)+": "+str(data[:,:,i].max())+" "+str(data[:,:,i].min())
    #     cv2.imwrite("test"+str(i)+".png",data[:,:,i])

    adj = adjacent(seed)

    output = gco.graph_cut(data,im_gray,seed,adj,num_labels)

    output = region_clean( region_shift(output,region_transform(output)) )

    # np.savetxt("image0041.label",output,fmt='%1d')
    np.savetxt("image0091.label",output,fmt='%1d')

    bmp_labels = label_to_bmp(output)
    cv2.imwrite('testoutput.png',draw_on_img(im,bmp_labels))

    return 0
 
if __name__ == '__main__':
    sys.exit(main(*sys.argv))

# im = np.asarray( im[:,:] )      # convert to numpy array
