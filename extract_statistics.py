#!/usr/bin/env python
import sys,os
sys.path.insert(0,os.path.join(os.getcwd(),'..'))
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt, binary_dilation
from scipy.ndimage import gaussian_filter, median_filter
import scipy
import cPickle as pickle
import matsciskel
import cv2
import itertools
import matsci.adj
import random
import matsci.gui

import matsci.gco

def edges(seed):
    grad = np.gradient(seed)
    return np.maximum(abs(grad[0]),abs(grad[1])) > 0

def histsamp(a, num):
    return np.searchsorted(np.cumsum(a),np.random.random(num))

def edge_list(seed):
    pairs = matsci.adj.Adj(seed).pairs()
    return zip(pairs,[np.logical_and(
                binary_dilation(seed==i),
                binary_dilation(seed==j)) 
                      for (i,j) in pairs ])

def main(*args):
    if(len(args) < 1):
        return 1

    import pylab

    center = []
    maxdeg = []
    mindeg = []
    counter = 0

    for i in range(1,len(args),2):
        im_path = args[i]
        gt_path = args[i+1]

        im = scipy.misc.imread(im_path,flatten=True).astype('float32')
        im = np.divide(im,im.max())
        im = np.multiply(im,255).astype('uint8')

        seed = np.genfromtxt(gt_path,dtype='int16')

        v = matsci.gco.Slice(im,seed)
        vs = v.local()

        for x in vs:
            center += [ np.sort(np.subtract(x.adj.local().sum(axis=0),1))[-1] ]
            maxdeg += [ np.sort(np.subtract(x.adj.local().sum(axis=0),1))[-2] ]
            mindeg += [ np.sort(np.subtract(x.adj.local().sum(axis=0),1))[0] ]
            if np.sort(np.subtract(x.adj.local().sum(axis=0),1))[-2] > 3:
                scipy.misc.imsave('deg/'+str(counter)+'.png',matsci.gui.color_jet(matsci.gui.grey_to_rgb(x.img),x.labels.v))
                counter += 1
            
    pylab.hist(center,align='left',bins=range(1,15))
    pylab.xlabel('local region degrees')
    pylab.ylabel('number of regions with degree')
    pylab.show()
    pylab.hist(maxdeg,align='left',bins=range(1,15))
    pylab.xlabel('degree of surrounding grain (max)')
    pylab.ylabel('number of regions with degree')
    pylab.show()
    pylab.hist(mindeg,align='left',bins=range(1,15))
    pylab.xlabel('degree of surrounding grain (min)')
    pylab.ylabel('number of regions with degree')
    pylab.show()

    # pickle.dump(center,open(os.path.basename(arg['label']).split('.')[0]+'-center.pkl','w'))
    # pickle.dump(maxdeg,open(os.path.basename(arg['label']).split('.')[0]+'-max.pkl','w'))
    # pickle.dump(mindeg,open(os.path.basename(arg['label']).split('.')[0]+'-min.pkl','w'))
    # print('Avg Center Deg: ' + str(np.mean(center)))
    # print('Avg Max Deg: ' + str(np.mean(maxdeg)))
    # print('Avg Min Deg: ' + str(np.mean(mindeg)))

    return 0

if __name__ == '__main__':
    sys.exit(main(*sys.argv))
