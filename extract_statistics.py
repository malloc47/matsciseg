#!/usr/bin/env python
import sys,os
sys.path.insert(0,os.path.join(os.getcwd(),'..'))
import numpy as np
import scipy
import cPickle as pickle
import matsciskel
# import matsci.adj
# import matsci.gui
# import matsci.gco
import matsci

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
            new_l = x.rev_shift(l)
            center += [ x.adj.deg(new_l,ignore_bg=True) ]
            ls = x.adj.degs(ignore_bg=True, ignore=[new_l])
            maxdeg += [max(ls,key=lambda x: x[1])[1]]
            mindeg += [min(ls,key=lambda x: x[1])[1]]

            if max(ls,key=lambda x: x[1])[1] == 6:
                scipy.misc.imsave('deg/'+str(counter)+'.png',
                                  highlight_label(
                        matsciskel.draw_on_img(matsci.gui.grey_to_rgb(x.img),
                                               matsciskel.label_to_bmp(x.labels.v))
                        , x.labels.v,
                        # center in red, deg>3 in blue
                        [(new_l, (128,0,0))] + \
                            [ (i, (0,0,128)) for (i,j) in ls if j == 4 ] + \
                            [ (i, (128,128,0)) for (i,j) in ls if j == 5 ] + \
                            [ (i, (128,0,128)) for (i,j) in ls if j == 6 ]
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

    return 0

if __name__ == '__main__':
    sys.exit(main(*sys.argv))
