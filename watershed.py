#!/usr/bin/env python
import sys,os
from matsci.data import watershed, watershed_fixed
from numpy import genfromtxt, savetxt
from scipy.misc import imread


def main(*args):
    if(len(args) < 3):
        return 1

    im = imread(args[1],flatten=True).astype('uint8')
    labels = genfromtxt(args[2],dtype='int16')

    label_out = args[3]

    suppression = int(3);
    dilation = int(0);

    if(len(args) > 4):
        suppression = int(args[4])

    if(len(args) > 5):
        dilation = int(args[5])

    w = watershed_fixed(im,labels,dilation,suppression)
    
    savetxt(label_out,w,fmt='%1d')

    return 0

if __name__ == '__main__':
    sys.exit(main(*sys.argv))
