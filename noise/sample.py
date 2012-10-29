#!/usr/bin/env python
import sys,os
sys.path.insert(0,os.path.join(os.getcwd(),'.'))
sys.path.insert(0,os.path.join(os.getcwd(),'..'))
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt, binary_dilation, grey_closing
from scipy.ndimage import gaussian_filter, median_filter
from scipy.stats import norm
import scipy
import cPickle as pickle
import matsciskel
import cv2
import itertools
import matsci.adj
import matsci.data
import matsci.label
import random
import math
from subprocess import check_output

def edges(seed):
    grad = np.gradient(seed)
    return np.maximum(abs(grad[0]),abs(grad[1])) > 0

def histsamp(a, num):
    return np.searchsorted(np.cumsum(a),np.random.random(num))

def expand_hist(a):
    return reduce(lambda i,j: i+j,[ [j]*i for (i,j) in zip(a,range(0,len(a))) ])

def normsamp(a, num, loc=None, scale=None):
    if loc is None or scale is None:
        loc, scale = norm.fit(a)
    return np.clip(norm.rvs(loc=loc, scale=scale, size=num),0,255)

def edge_list(seed):
    pairs = matsci.adj.Adj(seed).pairs()
    return zip(pairs,[np.logical_and(
                binary_dilation(seed==i),
                binary_dilation(seed==j)) 
                      for (i,j) in pairs ])

def line(c,a,l,shape):
    """
    create line from center point c, angle a, and length l in the
    specified square shape
    """
    p1 = tuple(map(int,map(round, [c[0] + l*math.cos(a), c[1] + l*math.sin(a)])))
    p2 = tuple(map(int,map(round, [c[0] + l*math.cos(a+math.pi), c[1] + l*math.sin(a+math.pi)])))
    return matsci.data.bresenham(p1+p2,shape)

def main(*args):
    if(len(args) < 3):
        return 1

    git_hash = check_output(['git','rev-parse','HEAD']).strip()

    for i in range(2,len(args),2):

        nlevels = 256

        seed = random.randint(0, sys.maxint)
        random.seed(seed)
        np.random.seed(seed)

        r = np.array(range(0,nlevels))
        source = pickle.load(open(args[1],'rb'))

        gt_path = args[i]
        out_path = args[i+1]
        seed_path = os.path.splitext(out_path)[0]+'.seed'
        ground = np.genfromtxt(gt_path,dtype='int16')
        ground_edges = edges(ground)

        with open(seed_path,'wb') as f:
            f.write(str(seed)+' '+git_hash)

        ncircles = random.randint(0,5)
        print('Number of circles: ' + str(ncircles))
        nlines = random.randint(0,2)
        print('Number of lines: ' + str(nlines))
        nscratches = random.randint(ground.min(),ground.max())
        print('Number of scratches: ' + str(nscratches))

        out = np.zeros(ground.shape,dtype='int16')

        dt = distance_transform_edt(np.logical_not(ground_edges))

        # supress edges
        for p,e in edge_list(ground):
            # lower threshold on edge "length"
            if e.sum() < 50:
                continue
            s = binary_dilation(e,iterations=3)
            dt[s] += random.choice([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.25, 0.5, 1])

#        for c in [ (random.randint(0,ground.shape[0]), 
#                    random.randint(0,ground.shape[1]), 
#                    random.randint(25,75)) 
#                   for k in range(0,ncircles) ]:
#            dt[matsci.adj.circle(c,out.shape)] += 1

        dt = dt.astype('int16')

        out[dt==0] = normsamp(source['dist_raw'][0], len(out[dt==0]), 255,128)

        out = grey_closing(out, size=(3,3))

        for d in source['dist']:
            if not d==0:
                out[dt==d] = histsamp(source['dist'][d], len(out[dt==d]))

        # fill in the rest of the pixels with value from last sampling
        m = max(source['dist'].keys())
        out[dt>=m-1] = histsamp(source['dist'][m], len(out[dt>=m-1]))

        # create varying intensities within grains
        for k in range(0,ground.max()):
            out[ground==k] += histsamp(source['grain_all'],1)[0]

        for l in [ binary_dilation(
                line( (random.randint(0,ground.shape[0]), 
                       random.randint(0,ground.shape[1]))
                      , random.random() * math.pi * 2 # angle
                      , int(math.sqrt(math.pow(ground.shape[0],2)+
                                      math.pow(ground.shape[1],2))) + 1 # length
                      , ground.shape))
                   for k in range(0,nlines) ]:
            out[l] += np.clip(map(int,map(round,norm.rvs(loc=64, 
                                                         scale=32, 
                                                         size=len(out[l]))))
                              ,0
                              ,255).astype('int16')

        # inter-grain scratches
        for k in random.sample(range(0,ground.max()), nscratches):
            reg = ground==k
            if reg.sum() < 500:
                continue
            c = scipy.ndimage.measurements.center_of_mass(reg)
            p = matsci.label.fit_region_z(reg)
            w = abs(p[0]-p[2])
            h = abs(p[1]-p[3])
            largest = int(round(min(w,h)/2))
            angle = random.random() * math.pi * 2
            val = random.randint(2,48)
            for j in range(1,random.randint(1,25)):
                l = np.logical_and(
                    line(
                        (random.randint(int(c[1]-h/2),
                                        int(c[1]+h/2)),
                         random.randint(int(c[0]-w/2),
                                        int(c[0]+w/2)))
                        , angle
                        , random.randint(2,max(3,largest))
                        , ground.shape)
                    , reg)
                out[l] += np.clip(
                    map(int,
                        map(round,
                            norm.rvs(
                                loc=val,
                                scale=32, 
                                size=len(out[l])))),
                    0,
                    255
                    ).astype('int16')
                
        out = np.clip(out,0,255).astype('uint8')

    # out = gaussian_filter(out, sigma=1)
    # out = median_filter(out, 3)

        cv2.imwrite(out_path,out)

    return 0

if __name__ == '__main__':
    sys.exit(main(*sys.argv))
