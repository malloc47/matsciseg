#!/usr/bin/env python
import os,sys
import numpy as np
import scipy
from scipy import ndimage
from matsci.draw import label_to_bmp, draw_on_img
from matsci.io import read_grey_as_rgb_unflattened, write_labels

def main(*args):
    if(len(args) < 4):
        return 1

    imgin = args[1]
    size = int(args[2])
    output = args[3]

    # im = read_grey_as_rgb(imgin)
    im = read_grey_as_rgb_unflattened(imgin)

    out = np.zeros((im.shape[0],im.shape[1]),dtype='int16')

    view = [np.array_split(o,size,axis=1) for o in np.array_split(out,size,axis=0)]

    counter = 0

    for i in range(0,len(view)):
        for j in range(0,len(view[i])):
            view[i][j][:] = counter
            counter += 1

    write_labels(out,output)

    return 0

if __name__ == '__main__':
    sys.exit(main(*sys.argv))
