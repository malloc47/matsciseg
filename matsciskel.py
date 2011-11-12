#!/usr/bin/env python
import sys,os,cv,cv2
import numpy as np
sys.path.insert(0,os.getcwd() + '/gco')
sys.path.insert(0,os.getcwd())
import gcoc,gco
import scipy
from scipy import ndimage

def display(im):
    cv2.namedWindow("tmp",cv2.CV_WINDOW_AUTOSIZE)
    cv2.imshow("tmp",im)
    cv2.waitKey(0)


def label_to_bmp(labels):
    grad = np.gradient(labels)
    seg = np.maximum(abs(grad[0]),abs(grad[1]))
    return seg

def draw_on_img(img,bmp):
    out = img.copy()
    out[np.nonzero(bmp>0)] = (0,0,255)
    return out

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

    v = gco.Volume(im_gray,seed)
    v.dilate(d)
    output = v.graph_cut()

    # np.savetxt("image0041.label",output,fmt='%1d')
    np.savetxt("image0091.label",output,fmt='%1d')

    bmp_labels = label_to_bmp(output)
    cv2.imwrite('testoutput.png',draw_on_img(im,bmp_labels))

    return 0
 
if __name__ == '__main__':
    sys.exit(main(*sys.argv))

# im = np.asarray( im[:,:] )      # convert to numpy array
