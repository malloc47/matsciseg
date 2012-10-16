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

def highlight_label(img,labels,h):
    img = img.astype('float')
    for (l,c) in h:
        r = img[:,:,0]
        g = img[:,:,1]
        b = img[:,:,2]
        r[labels==l] += c[0]
        g[labels==l] += c[1]
        b[labels==l] += c[2]
    return np.clip(img,0,255).astype('uint8')

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

        for x,l in vs:
            ls = zip(range(1,len(x.adj.deg(ignore_background=True))),
                     x.adj.deg(ignore_background=True))
            new_l = x.rev_shift(l)
            center += [ j for (i,j) in ls if i == new_l ]
            ls = [ (i,j) for (i,j) in ls if i != new_l ]
            maxdeg += [max(ls,key=lambda x: x[1])[1]]
            mindeg += [min(ls,key=lambda x: x[1])[1]]

            # ls = np.delete(x.adj.deg(ignore_background=True),new_l-1)
            # center += [x.adj.deg(new_l,ignore_background=True)]
            # maxdeg += [max(ls)]
            # mindeg += [min(ls)]
            # center += [ np.sort(np.subtract(x.adj.local().sum(axis=0),1))[-1] ]
            # maxdeg += [ np.sort(np.subtract(x.adj.local().sum(axis=0),1))[-2] ]
            # mindeg += [ np.sort(np.subtract(x.adj.local().sum(axis=0),1))[0] ]
            # if np.sort(np.subtract(x.adj.local().sum(axis=0),1))[-2] > 3:
            if max(ls,key=lambda x: x[1])[1] > 3:
                # scipy.misc.imsave('deg/'+str(counter)+'.png',matsci.gui.color_jet(matsci.gui.grey_to_rgb(x.img),x.labels.v))
                scipy.misc.imsave('deg/'+str(counter)+'.png',
                                  highlight_label(
                        matsciskel.draw_on_img(matsci.gui.grey_to_rgb(x.img),
                                               matsciskel.label_to_bmp(x.labels.v))
                        , x.labels.v,
                        # center in red, deg>3 in blue
                        [(new_l, (128,0,0))] + [ (i, (0,0,128)) for (i,j) in ls if j > 3 ]
                        ))
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
