#!/usr/bin/env python
import sys,os
import numpy as np
import matsciskel
import matsci.gco

def main(*args):
    if(len(args) < 4):
        return 1

    for i in range(1,len(args),3):
        imgin = args[i];
        label_path = args[i+1];
        out_path = args[i+2];
        im,im_gray = matsciskel.read_img(imgin)
        seed=np.genfromtxt(label_path,dtype='int16')
        v = matsci.gco.Slice(im_gray,seed)
        v.save(out_path)

    return 0

if __name__ == '__main__':
    sys.exit(main(*sys.argv))
