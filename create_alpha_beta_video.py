#!/usr/bin/env python
import os,sys
import numpy as np
import scipy
from scipy import ndimage
import render_labels

def convert_linear(linear,w,h):
    output = np.zeros((h,w),dtype='int16')
    counter = 0
    for i in range(h):
        for j in range(w):
            output[i,j] = int(linear[counter])
            counter += 1
    return output

def main(*args):
    if(len(args) < 3):
        return 1

    w = int(args[1])
    h = int(args[2])
    imgin = args[3]

    im = scipy.misc.imread(imgin,flatten=True).astype('float32')
    im = np.divide(im,im.max())
    im = np.dstack((im,im,im))
    im = np.multiply(im,255).astype('uint8')

    # im = np.hstack((im,im))

    for fname in args[4:]:
        f = open(fname, 'r')
        label = convert_linear(f.read().split(),w,h)
        fn = os.path.basename(fname)
        num = int(fn[5:-12].split('-')[0])
        l1 = int(fn[5:-12].split('-')[1])
        l2 = int(fn[5:-12].split('-')[2])
        # print(str(im))
        # print(str(im.shape))
        # print(str(label))
        # print(str(label.shape))
        im_new = render_labels.draw_on_img(im.copy(),render_labels.label_to_bmp(label))
        im2 = render_labels.draw_on_img(im_new.copy(),label==l1,(0,255,0))
        im2 = render_labels.draw_on_img(im2,label==l2,(0,0,255))
        im_new = np.hstack((im2,im_new))
        scipy.misc.imsave(str(num)+'.png',im_new);
        
    

    # output = np.zeros((h,w),dtype='int16')

    # counter = 0
    # print(str(len(label)))
    # for i in range(h):
    #     for j in range(w):
    #         output[i,j] = int(label[counter])
    #         counter += 1
            
    # np.savetxt(out,output,fmt='%1d')    

    return 0

if __name__ == '__main__':
    sys.exit(main(*sys.argv))
