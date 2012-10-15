#!/usr/bin/env python
import sys,os
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt, binary_erosion
import cPickle as pickle
import matsciskel

def edges(seed):
    grad = np.gradient(seed)
    return np.maximum(abs(grad[0]),abs(grad[1])) > 0

def main(*args):
    if(len(args) < 3):
        return 1

    nlevels = 256
    maxdist = 80
    minsamp = 3000
    
    r = np.array(range(0,nlevels))
    dist_hists = {}
    grain_hist = np.zeros(256,dtype='int16')

    for i in range(1,len(args),2):
        im_path = args[i]
        gt_path = args[i+1]

        print(gt_path)

        im,img = matsciskel.read_img(im_path)

        ground=np.genfromtxt(gt_path,dtype='int16')
        ground_edges=edges(ground)

        dt = distance_transform_edt(np.logical_not(ground_edges)).astype('int')

        for j in range(0,maxdist):
            tmp = img[dt==j]
            b = np.histogram(tmp,bins=nlevels)[0]
            if len(tmp) >= minsamp:
                if j in dist_hists:
                    dist_hists[j] += b
                else:
                    dist_hists[j] = b
            else:
                break

        for j in range(0,ground.max()):
            try:
                grain_hist[int(np.mean(img[binary_erosion(ground==j,iterations=7)]))] += 1
            except:
                pass
                # print("Didn't get region")

    # normalize everything
    for d in dist_hists:
        dist_hists[d] = dist_hists[d].astype('float64')
        dist_hists[d] /= sum(dist_hists[d])

    grain_hist = grain_hist.astype('float64')
    grain_hist /= sum(grain_hist)

    pickle.dump(dist_hists,open('histograms.pkl','wb'))
    pickle.dump(grain_hist,open('grains.pkl','wb'))

    return 0

if __name__ == '__main__':
    sys.exit(main(*sys.argv))
