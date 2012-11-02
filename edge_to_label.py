#!/usr/bin/env python
import os,sys
import numpy as np
import scipy
from scipy.ndimage.measurements import label
from matsci.label import small_filter, region_clean

def main(*args):
    if(len(args) < 2):
        return 1

    im = scipy.misc.imread(args[1],flatten=True).astype('float32')
    im = np.divide(im,im.max())
    im = np.multiply(im,255).astype('uint8')
    im = (im > 0)

    labels = region_clean(
        small_filter(
            label(
                np.logical_not(im)
                )[0]
            , 0)) - 1

    np.savetxt(args[2],labels,fmt='%1d')
    return 0

if __name__ == '__main__':
    sys.exit(main(*sys.argv))
