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

def read_img(img_name):
    im = cv.LoadImageM(img_name)
    im_gray = cv.CreateMat(im.rows, im.cols, cv.CV_8U)
    cv.CvtColor(im,im_gray, cv.CV_RGB2GRAY)
    # convert to numpy arrays
    im_gray = np.asarray(im_gray[:,:])
    im = np.asarray(im[:,:])
    return (im,im_gray)

def main(*args):
    d = 10
    if(len(args) < 6):
        return 1

    arg = {'im1'       : args[1],
           'label'    : args[2],
           'im2'       : args[3],
           'label_out' : args[4],
           'im_out'    : args[5]}

    im,im_gray = read_img(arg['im2'])
    seed=np.genfromtxt(arg['label'],dtype='int16')

    v = gco.Volume(im_gray,seed)
    print("Initialized")
    v.dilate(d)
    v.dilate_first(d/2)
    print("Dilated")
    v.skel()
    output = v.graph_cut()

    # np.savetxt("image0041.label",output,fmt='%1d')
    np.savetxt(arg['label_out'],output,fmt='%1d')

    bmp_labels = label_to_bmp(output)
    cv2.imwrite(arg['im_out'],draw_on_img(im,bmp_labels))

    return 0
 
if __name__ == '__main__':
    sys.exit(main(*sys.argv))
