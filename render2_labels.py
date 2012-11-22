#!/usr/bin/env python
import os,sys
import numpy as np
import scipy
from scipy import ndimage

def draw_on_img(img,bmp,color=(255,0,0)):
    out = img.copy()
    out[np.nonzero(bmp>0)] = color
    return out

def alpha_blend(img,bmp,color=(255,0,0,0.5)):
    out = img.copy().astype('int16')
    out[np.nonzero(bmp>0)] = np.add(out[np.nonzero(bmp>0)],tuple([c*color[-1] for c in color[:-1]]))
    return np.clip(out,0,255).astype('uint8')

def label_to_bmp(labels):
    grad = np.gradient(labels)
    seg = np.maximum(abs(grad[0]),abs(grad[1]))
    return seg

def read_grey_as_rgb(imgin):
    im = scipy.misc.imread(imgin,flatten=True).astype('float32')
    im = np.divide(im,im.max())
    im = np.multiply(im,255)
    im = np.dstack((im,im,im))
    return im

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
