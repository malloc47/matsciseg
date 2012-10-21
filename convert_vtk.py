#!/usr/bin/env python
import os,sys
import numpy as np
import scipy
from scipy import ndimage
from matsci.gui import color_jet
from render_labels import draw_on_img, label_to_bmp

def main(*args):
    if(len(args) < 3):
        return 1

    vtk_path = args[1];
    output_file = args[2];

    size = (525,750,50)
    i = 0
    j = 0
    k = 0

    output = np.zeros(size,dtype='int16')

    with open(vtk_path) as f:
        # skip over front matter
        line = f.readline()
        while not line.startswith('LOOKUP_TABLE'):
            # print('skipping: ' + line)
            line = f.readline()
        for line in f:
            labels = line.split()
            for l in labels:
                # output[i] = l
                # i += 1
                output[i,j,k] = l
                j += 1
                if j > size[1]-1:
                    j=0
                    i+=1
                if i > size[0]-1:
                    i=0
                    j=0
                    k+=1

    img = np.zeros((size[0],size[1],3), dtype='uint8')

    # for i in range(0,size[0]):
    #     for j in range(0,size[1]):
    #         for k in range(0,size[2]):
    #             output2[i,j,k] = output[ k*(size[1]*size[0]) + j*size[0] + i ]

    for k in range(0,size[2]):
        np.savetxt(output_file+format(k,'04d')+'.label',output[:,:,k],fmt='%1d')
        scipy.misc.imsave(output_file+format(k,'04d')+'.png',
                          draw_on_img(color_jet(img,
                                                output[:,:,k],
                                                alpha=1.0), 
                                      label_to_bmp(output[:,:,k])))
        
    return 0

if __name__ == '__main__':
    sys.exit(main(*sys.argv))
