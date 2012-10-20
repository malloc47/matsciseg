#!/usr/bin/env python
import sys,os
sys.path.insert(0,os.path.join(os.getcwd(),'..'))
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
    output = {}
    output['dist'] = {}
    output['dist_raw'] = {}
    output['grain_all'] = np.zeros(256,dtype='int16')

    for i in range(1,len(args),2):
        im_path = args[i]
        gt_path = args[i+1]

        print(gt_path)

        im,img = matsciskel.read_img(im_path)

        ground=np.genfromtxt(gt_path,dtype='int16')
        ground_edges=edges(ground)

        dt = distance_transform_edt(np.logical_not(ground_edges)).astype('int')

        output['dist'][0] = np.histogram(img[ground_edges],bins=nlevels)[0]
        output['dist_raw'][0] = img[ground_edges]

        for j in range(1,maxdist):
            tmp = img[dt==j]
            output['dist_raw'][j] = tmp
            b = np.histogram(tmp,bins=nlevels)[0]
            if len(tmp) >= minsamp:
                if j in output['dist']:
                    output['dist'][j] += b
                else:
                    output['dist'][j] = b
            else:
                break

        for j in range(0,ground.max()):
            try:
                output['grain_all'][int(np.mean(img[binary_erosion(ground==j,iterations=7)]))] += 1
            except:
                pass
                # print("Didn't get region")

    # normalize everything
    for d in output['dist']:
        output['dist'][d] = output['dist'][d].astype('float64')
        output['dist'][d] /= sum(output['dist'][d])

    output['grain_all'] = output['grain_all'].astype('float64')
    output['grain_all'] /= sum(output['grain_all'])

    pickle.dump(output['dist'],open('histograms.pkl','wb'))
    pickle.dump(output['grain_all'],open('grains.pkl','wb'))
    pickle.dump(output,open('data.pkl','wb'))

    return 0

if __name__ == '__main__':
    sys.exit(main(*sys.argv))
