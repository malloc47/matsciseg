#!/usr/bin/env python
import sys,os
import cPickle as pickle
from numpy import genfromtxt
import matsciskel
import gco

def main(*args):
    if(len(args) < 5):
        return 1

    slices = {}

    for i in range(2,len(args),3):
        n = int(args[i])
        imgin = args[i+1];
        label_path = args[i+2];
        print('Pickling slice ' + str(n))
        im,im_gray = matsciskel.read_img(imgin)
        seed=genfromtxt(label_path,dtype='int16')
        v = gco.Slice(im_gray,seed)
        slices[n] = v

    pickle.dump(slices,open(args[1],'wb'))

    return 0

if __name__ == '__main__':
    sys.exit(main(*sys.argv))
