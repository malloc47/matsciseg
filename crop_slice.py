#!/usr/bin/env python
import os,sys
import numpy as np
import scipy
from scipy import ndimage
from matsci.gco import region_transform, region_clean, region_shift

def main(*args):

    if(len(args) < 8):
        return 1

    label_path = args[1]
    imgin = args[2]
    top = int(args[3])
    left = int(args[4])
    width = int(args[5])
    height = int(args[6])
    labelout = args[7]
    imout = args[8]

    im = scipy.misc.imread(imgin,flatten=True)
    labels = np.genfromtxt(label_path,dtype='int16')

    imo = im[top:top+height,left:left+width]
    labelso = labels[top:top+height,left:left+width]
    labelso = region_clean(region_shift(labelso, region_transform(labelso)))

    scipy.misc.imsave(imout,imo);
    np.savetxt(labelout,labelso,fmt='%1d')

    return 0

if __name__ == '__main__':
    sys.exit(main(*sys.argv))
