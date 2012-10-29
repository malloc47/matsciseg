#!/usr/bin/env python
import os,sys
import numpy as np
import scipy
import scipy.ndimage
#from matsci.gui import color_jet
from render_labels import draw_on_img, label_to_bmp

def label_preprocess(labels):
    new_label = labels.max()+1
    for l in range(labels.max()+1):
        layer = (labels==l)

        if not layer.any():
            continue

        ls,n = scipy.ndimage.label(layer)

        if n < 2:
            continue
        
        for m in range(2,n+1):
            labels[ls == m] = new_label
            new_label += 1

    return labels

def main(*args):
    if(len(args) < 3):
        return 1

    vtk_path = args[1];
    output_file = args[2];

    size = (750,525,500)
    i = 0
    j = 0
    k = 0

    output = np.zeros(size,dtype='int16')

    print('reading vtk file')
    with open(vtk_path) as f:
        # skip over front matter
        line = f.readline()
        while not line.startswith('DIMENSIONS'):
            line = f.readline()
        x,y,z = [ i-1 for i in map(int,line.split(' ')[1:]) ]
        size = (y,x,z)
        output = np.zeros(size,dtype='int16')
        while not line.startswith('LOOKUP_TABLE'):
            line = f.readline()
        for line in f:
            labels = line.split()
            for l in labels:
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

    print('preprocessing')
    for k in range(0,size[2]):
        output[:,:,k] = label_preprocess(output[:,:,k])

    print('writing out slices')
    for k in range(0,size[2]):
        np.savetxt(output_file+format(k,'04d')+'.label',
                   output[:,:,k],fmt='%1d')
        # scipy.misc.imsave(output_file+format(k,'04d')+'.png',
        #                   draw_on_img(color_jet(img,
        #                                         output[:,:,k],
        #                                         alpha=1.0), 
        #                               label_to_bmp(output[:,:,k])))
        
    return 0

if __name__ == '__main__':
    sys.exit(main(*sys.argv))
