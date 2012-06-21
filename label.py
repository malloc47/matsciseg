# label module
import numpy as np
import scipy
from scipy import ndimage

def create_mask(labels,label_list):
    """get binary mask from a list of labels (in an integer matrix)"""
    return reduce(np.logical_or,map(lambda l:labels==l,label_list))

def fit_region(im):
    """return coordinates of box that fits around a binary region"""
    mask_win = np.argwhere(im)
    (y0, x0), (y1, x1) = mask_win.min(0), mask_win.max(0) + 1 
    return (x0,y0,x1,y1)

def label_max(labels):
    """return index of largest label"""
    sizes = []
    # check only indices above 0
    for l in range(1,labels.max()+1) :
        sizes.append((labels == l).sum())
    # increment index, as we ignored zero before
    return sizes.index(max(sizes))+1

def label_sizes(labels):
    """return sizes of all labels in an integer matrix"""
    return map(lambda l:(labels==l).sum(),range(0,labels.max()+1))

def label_empty(labels):
    """return labels that are empty (size==0)"""
    return [i for i,x in enumerate(label_sizes(labels)) if x == 0]

def region_transform(labels):
    """find regions that are empty and compact them down"""
    unshifted = range(0,labels.max()+1)
    shifted = range(0,labels.max()+1)
    counter = 0
    empty = label_empty(labels)
    for l in unshifted :
        if l in empty :
            shifted[l] = -1
        else:
            shifted[l] = counter
            counter = counter + 1
    return zip(unshifted,shifted)

def region_shift(regions,transform) :
    """change labels based on order pairs"""
    labels = np.zeros(regions.shape,dtype=regions.dtype)
    for t in transform :
        if (t[0] < 0 or t[1] < 0) : continue
        labels[regions == t[0]] = t[1]
    return labels

def largest_connected_component(im):
    labels,num = ndimage.label(im)
    if num < 1:
        return im
    sizes = [ ((labels==l).sum(),l) for l in range(1,num+1) ]
    return labels==max(sizes,key=lambda x:x[0])[1]

def small_filter(labels,label_num):
    """cover small label region with the nearest label"""
    mask = create_mask(labels,[label_num]);
    if not mask.any():
        return labels
    (x0,y0,x1,y1) = fit_region_z(mask);
    if (x0>0): x0 -= 1;
    if (y0>0): y0 -= 1;
    if (x1<labels.shape[1]-1): x1 += 1;
    if (y1<labels.shape[0]-1): y1 += 1;
    chk_win = mask[y0:y1+1,x0:x1+1];

    inds = scipy.ndimage.morphology.distance_transform_edt(chk_win,
                                                           return_distances=False,
                                                           return_indices=True);

    for y in range(0,chk_win.shape[0]):
        for x in range(0,chk_win.shape[1]):
            labels[y+y0,x+x0] = labels[inds[0][y,x]+y0,inds[1][y,x]+x0]

    return labels

def region_clean(regions):
    out = np.ones(regions.shape,dtype=regions.dtype)*-1
    for l in range(regions.max()+1) :
        layer = (regions==l)
        if layer.any() :
            # labeled = ndimage.label(layer>0)[0]
            labeled = largest_connected_component(layer>0)
            # copy only the largest connected component
            out[np.nonzero(labeled == label_max(labeled))] = l
    return small_filter(out,-1)
    # todo: what's happening with the -1s?
    # return out

def fit_region_z(im):
    """similar to fit_region"""
    mask_win = np.argwhere(im)
    (y0,x0),(y1,x1) = mask_win.min(0),mask_win.max(0)
    return (x0,y0,x1,y1)

class Label(object):
    def __init__(self,labels=None):
        if not labels is None:
            self.v = region_clean(region_shift(labels,
                                               region_transform(labels)))
    
    def create_mask(self,label_list):
        """get binary mask from a list of labels (in an integer matrix)"""
        return reduce(np.logical_or,map(lambda l:self.v==l,label_list))

    def max(self):
        return self.v.max()

    