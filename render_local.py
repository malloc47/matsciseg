#!/usr/bin/env python
import os,sys
import numpy as np
import scipy
from scipy import ndimage
import matsci
from matsci.gui import jet_colors, randmap
import matsciskel
import random

def alpha_blend(img,bmp,color=(255,0,0,0.5)):
    out = img.copy()
    out[np.nonzero(bmp>0)] = np.add(out[np.nonzero(bmp>0)],tuple([c*color[-1] for c in color[:-1]]))
    return out

def plot_that_prints_out_values_when_clicked(labels):
    from matplotlib import pyplot as plt
    import numpy as np

    im = plt.imshow(labels, interpolation='nearest')
    fig = plt.gcf()
    ax = plt.gca()

    class EventHandler:
        def __init__(self):
            fig.canvas.mpl_connect('button_press_event', self.onpress)

        def onpress(self, event):
            if event.inaxes!=ax:
                return
            xi, yi = (int(round(n)) for n in (event.xdata, event.ydata))
            value = im.get_array()[yi,xi]
            color = im.cmap(im.norm(value))
            print value

    handler = EventHandler()

    plt.show()

def main(*args):
    if(len(args) < 4):
        return 1

    seed = random.randint(0, sys.maxint)
    random.seed(seed)
    np.random.seed(seed)
    print(str(seed))

    label_path = args[2];
    imgin = args[1];
    output = args[3];
    dilation = 5
    alpha = 0.5

    im = scipy.misc.imread(imgin,flatten=True).astype('float32')
    im = np.divide(im,im.max())
    im_gray = np.multiply(im,255).astype('uint8')
    im = np.dstack((im_gray,im_gray,im_gray))
    labels = np.genfromtxt(label_path,dtype='int16')
    bmp = matsciskel.label_to_bmp(labels)
    # if(len(args) > 4):
    #     bmp = scipy.ndimage.morphology.binary_dilation(bmp,iterations=int(args[4]))

    v = matsci.gco.Slice(im_gray,labels)

    out = (im.copy()).astype('int16')

    # for l in [16,25,33,47,81,55,94]: # random.sample((0,labels.max()+1),5):
    for l in [16,33,47]: # random.sample((0,labels.max()+1),5):
        a = v.adj.get_adj([l])
        mask = scipy.ndimage.morphology.binary_dilation(
            matsci.label.binary_remove(v.labels.create_mask(a))
            , iterations=5)
        color=jet_colors[randmap[l%128]]+(alpha,)
        out = alpha_blend(alpha_blend(out,mask,color=color)
                      , labels==l
                      , color=color)

    scipy.misc.imsave(output,np.clip(out,0,255).astype('uint8'))

    return 0

if __name__ == '__main__':
    sys.exit(main(*sys.argv))
