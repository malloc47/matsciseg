#!/usr/bin/env python
import os,sys
import numpy as np
import scipy
from scipy import ndimage

import matplotlib as mpl
mpl.use('svg')
import matplotlib.pyplot as plt
# import pylab as plt

import matsciskel
from matsci.label import pts_to_label
from matsci.adj import Adj


def main(*args):
    if(len(args) < 3):
        return 1

    try:
        markersize = args[4]
    except:
        markersize = 25

    try:
        linewidth = args[5]
    except:
        linewidth = 5

    im,im_gray =  matsciskel.read_img(args[1])
    if args[2].endswith('.label'):
        labels = np.genfromtxt(args[2],dtype='int16')
        centers = [ (t[1],t[0]) for t in
                    ndimage.measurements.center_of_mass(labels,
                                                        labels,
                                                        range(0,labels.max()+1))]
    else:
        pts = scipy.misc.imread(args[2],flatten=True).astype('bool')
        lbl,num = ndimage.label(pts)
        labels = pts_to_label(pts)
        centers = [ (t[1],t[0]) for t in 
                    ndimage.measurements.center_of_mass(pts,lbl,range(1,num+1))]
    adj = Adj(labels).pairs()

    fig = plt.figure()
    plt.axis('off')
    ax=fig.add_subplot(111)
    img = plt.imshow(im)
    dpi = fig.get_dpi()
    fig.set_size_inches(im_gray.shape[1]/dpi,im_gray.shape[0]/dpi)
    plt.subplots_adjust(top=1, right=1, bottom=0.0, left=0.0, wspace=0, hspace=0)
    # extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())

    # plot lines
    for p in adj:
        x = [ centers[p[0]][0] , centers[p[1]][0] ]
        y = [ centers[p[0]][1] , centers[p[1]][1] ]
        plt.plot(x,y,color='r',ls='--',linewidth=linewidth)
    # plot points
    ax.plot(*zip(*centers),marker='o', color='r', ls='', markersize=markersize)

    fig.savefig(args[3],bbox_inches='tight',pad_inches=0)
    

if __name__ == '__main__':
    sys.exit(main(*sys.argv))
