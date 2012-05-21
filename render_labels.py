#!/usr/bin/env python
import os,sys
import numpy as np
import scipy
from scipy import ndimage

def draw_on_img(img,bmp):
    out = img.copy()
    out[np.nonzero(bmp>0)] = (255,0,0)
    return out

def label_to_bmp(labels):
    grad = np.gradient(labels)
    seg = np.maximum(abs(grad[0]),abs(grad[1]))
    return seg

def main(*args):
    if(len(args) < 3):
        return 1

    label_path = args[1];
    imgin = args[2];
    output = args[3];

    im = scipy.misc.imread(imgin,flatten=True)
    im = np.dstack((im,im,im))
    labels = np.genfromtxt(label_path,dtype='int16')
    bmp = label_to_bmp(labels)
    if(len(args) > 3):
        bmp = scipy.ndimage.morphology.binary_dilation(bmp,iterations=int(args[4]))
    scipy.misc.imsave(output,draw_on_img(im,bmp));

    return 0

if __name__ == '__main__':
    sys.exit(main(*sys.argv))