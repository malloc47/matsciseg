#!/usr/bin/env python
import os,sys
import numpy as np
import scipy
from scipy import ndimage

def main(*args):
    if(len(args) < 3):
        return 1

    w = int(args[1])
    h = int(args[2])
    f = open(args[3], 'r')
    label = f.read().split()
    out = args[4]

    output = np.zeros((h,w),dtype='int16')

    counter = 0

    print(str(len(label)))
    for i in range(h):
        for j in range(w):
            output[i,j] = int(label[counter])
            counter += 1
            
    np.savetxt(out,output,fmt='%1d')    

    return 0

if __name__ == '__main__':
    sys.exit(main(*sys.argv))
