#!/usr/bin/env python
import os,sys
import numpy as np
import scipy
from scipy import ndimage
from matsci.draw import label_to_bmp, draw_on_img
from matsci.io import read_grey_as_rgb

def main(*args):
    if(len(args) < 4):
        return 1

    label_path = args[2];
    imgin = args[1];
    output = args[3];

    # im = scipy.misc.imread(imgin,flatten=True).astype('float32')
    # im = np.divide(im,im.max())
    # im = np.multiply(im,255).astype('uint8')
    # im = np.dstack((im,im,im))
    im = read_grey_as_rgb(imgin)
    labels = np.genfromtxt(label_path,dtype='int16')
    bmp = label_to_bmp(labels)
    if(len(args) > 4):
        bmp = scipy.ndimage.morphology.binary_dilation(bmp,iterations=int(args[4]))
    scipy.misc.imsave(output,draw_on_img(im,bmp));

    return 0

if __name__ == '__main__':
    sys.exit(main(*sys.argv))
