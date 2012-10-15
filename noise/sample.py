#!/usr/bin/env python
import sys,os
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt, binary_dilation
from scipy.ndimage import gaussian_filter, median_filter
import cPickle as pickle
import matsciskel
import cv2
import itertools
import matsci.adj
import random

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
    if(len(args) < 3):
        return 1

    nlevels = 256

    r = np.array(range(0,nlevels))
    hists = pickle.load(open(args[1],'rb'))
    grain_hist = pickle.load(open(args[2],'rb'))

    out = None

    # create edges
    for i in range(3,len(args),2):
        gt_path = args[i]
        out_path = args[i+1]
        ground = np.genfromtxt(gt_path,dtype='int16')
        ground_edges = edges(ground)

        if out is None:
            out = np.zeros(ground.shape,dtype='int16')

        dt = distance_transform_edt(np.logical_not(ground_edges))

        for p,e in edge_list(ground):
            s = binary_dilation(e,iterations=3)
            dt[s] += random.choice([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.25, 0.5, 1])

        dt = dt.astype('int16')

        for d in hists:
            out[dt==d] = histsamp(hists[d], len(out[dt==d]))

        # fill in the rest of the pixels with value from last sampling
        m = max(hists.keys())
        out[dt>=m-1] = histsamp(hists[m], len(out[dt>=m-1]))

    # out = np.clip(out,0,255)

    # create varying intensities within grains
    for i in range(0,ground.max()):
        out[ground==i] += histsamp(grain_hist,1)[0]

    out = np.clip(out,0,255).astype('uint8')
    # out = gaussian_filter(out, sigma=1)
    # out = median_filter(out, 3)

    cv2.imwrite(out_path,out)

    return 0

if __name__ == '__main__':
    sys.exit(main(*sys.argv))
