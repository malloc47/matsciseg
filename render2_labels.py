#!/usr/bin/env python
import os,sys
import numpy as np
import scipy
from scipy import ndimage
from matsci.draw import alpha_blend, label_to_bmp, draw_on_img
from matsci.io import read_grey_as_rgb

def main(*args):
    if(len(args) < 4):
        return 1

    label_path = args[2]
    label2_path = args[3]
    imgin = args[1]
    output = args[4]

    im = read_grey_as_rgb(imgin)
    labels = np.genfromtxt(label_path,dtype='int16')
    labels2 = np.genfromtxt(label2_path,dtype='int16')
    bmp = label_to_bmp(labels)
    if(len(args) > 5):
        bmp = scipy.ndimage.morphology.binary_dilation(bmp,iterations=int(args[5]))
    scipy.misc.imsave(output,
                      alpha_blend(
            draw_on_img(im,bmp,color=(0,0,255))
            , label_to_bmp(labels2)
            , color=(255,0,0,0.5)))

    return 0

if __name__ == '__main__':
    sys.exit(main(*sys.argv))
