# label module
import numpy as np
import scipy
from scipy import ndimage
from scipy.ndimage.morphology import binary_dilation
import data,adj

def create_mask(labels,label_list):
    """get binary mask from a list of labels (in an integer matrix)"""
    return reduce(np.logical_or,map(lambda l:labels==l,label_list))

def binary_remove(img):
    return data.relative_complement((img,
                                     scipy.ndimage.generic_filter(img,
                                                                  lambda d: d.all(),
                                                                  mode='constant',
                                                                  cval=0,
                                                                  footprint=[[0,1,0],[1,1,1],[0,1,0]])))

def region_outline(labels):
    indices = binary_remove(labels > 0).nonzero()
    # convert to list of tuples with indices and labels
    return [(i,j,labels[i,j]) for i,j in zip(indices[0].tolist(),indices[1].tolist())]

def centers_of_mass(labels):
    return [ (int(i),int(j),l) for i,j,l in
             [ ndimage.measurements.center_of_mass(labels==l) + (l,)
              for l in range(0,labels.max()+1)]
             if not np.isnan(i) and not np.isnan(j)]

def region_boundary_intensity(labels,img,l,t):
    boundary = img[binary_remove(labels==l)]
    thresh = np.count_nonzero(boundary > t)
    total = len(boundary)
    return float(thresh)/float(total)

def fit_region(im, padding=0):
    """return coordinates of box that fits around a binary region"""
    mask_win = np.argwhere(data.dilate(im,padding)) if padding > 0 \
        else np.argwhere(im)
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

def num_components(labels,full=True):
    # import scipy.misc
    # test = [ ndimage.label(labels==l
    #                        ,structure=[[1,1,1],[1,1,1],[1,1,1]])[1] 
    #          for l in range(labels.max()+1) ]
    # for i in range(len(test)):
    #     if test[i] > 1:
    #         output = (labels == i)
    #         output[output==False] = 0
    #         output[output==True] = 255
    #         scipy.misc.imsave("comp"+str(i)+".png",output)
    return [ ndimage.label(
            labels==l,
            structure=[[1,1,1],[1,1,1],[1,1,1]] 
            if full else
            [[0,1,0],[1,1,1],[0,1,0]])[1] 
             for l in range(labels.max()+1) ]

def boundary_connected_component(im,boundary):
    labels,num = ndimage.label(im)
    if num < 1:
        return im
    sizes_connections = [ ((labels==l).sum()
                           , (np.logical_and(binary_dilation(labels==l), 
                                            boundary)).sum()
                           , l)
                          for l in range(1,num+1) ]
    # print(str(sizes_connections))
    connected = [ (s,c,l) for (s,c,l) in sizes_connections
                  if c == max(sizes_connections, key=lambda x: x[1])[1] ]

    if len(connected) < 2:
        return labels == connected[0][2]
    else:
        return labels == max(connected, key=lambda x: x[0])[2]

def small_filter(labels,label_num):
    """cover small label region with the nearest label"""
    mask     = create_mask(labels,[label_num]);
    mask_err = np.argwhere(mask);

    (x0,y0) = (0,0);
    (y1,x1) = mask.shape;
    x1 -= 1;
    y1 -= 1;

    inds = scipy.ndimage.morphology.distance_transform_edt(mask,
                                                           return_distances=False,
                                                           return_indices=True);

    for err in mask_err:
        y = err[0];
        x = err[1];
        labels[y,x] = labels[inds[0][y,x],inds[1][y,x]];

    return labels

def region_clean(regions, boundary=None,bg=False):
    out = np.ones(regions.shape,dtype=regions.dtype)*-1
    if bg:
        out[regions==0] = 0
    for l in range(regions.max()+1) if not bg else range(1,regions.max()+1) :
        layer = (regions==l)
        if layer.any() :
            labeled = largest_connected_component(layer>0)
            # if boundary is None:
            #     labeled = largest_connected_component(layer>0)
            # else:
            #     labeled = boundary_connected_component(layer>0, boundary)
            # copy only the largest connected component
            out[np.nonzero(labeled == label_max(labeled))] = l
    return small_filter(out,-1)
    # return out

def junctions(regions,d=3):
    ps = np.nonzero(scipy.ndimage.generic_filter(regions,
                                 lambda a: len(np.unique(a)) >= 3,
                                 mode='nearest',
                                 size=(2,2)))
    ps = [ (i,j,3) for (i,j) in zip(ps[0],ps[1]) ]
    return [ (p[:-1] , set(np.unique(regions[adj.circle(p,regions.shape)])) )
             for p in ps ]

def fit_region_z(im):
    """similar to fit_region"""
    mask_win = np.argwhere(im)
    (y0,x0),(y1,x1) = mask_win.min(0),mask_win.max(0)
    return (x0,y0,x1,y1)

def edge_list(labels):
    pairs = matsci.adj.Adj(labels).pairs()
    return zip(pairs,[np.logical_and(
                binary_dilation(labels==i),
                binary_dilation(labels==j)) 
                      for (i,j) in pairs ])

class Label(object):
    def __init__(self,labels=None, boundary=None, bg=False):
        if not labels is None:
            self.v = region_clean(region_shift(labels
                                               , region_transform(labels))
                                  , boundary=boundary
                                  , bg=bg)
    
    def create_mask(self,label_list):
        """get binary mask from a list of labels (in an integer matrix)"""
        return reduce(np.logical_or,map(lambda l:self.v==l,label_list))

    def max(self):
        return self.v.max()

    def sizes(self):
        return label_sizes(self.v)

    def region_outline(self):
        return region_outline(self.v)

    def centers_of_mass(self):
        return centers_of_mass(self.v)

    def junctions(self,d=3):
        return junctions(self.v,d)

    def region_boundary_intensity(self,img,l,t):
        return region_boundary_intensity(self.v,img,l,t)

    def num_components(self,full=True):
        return num_components(self.v,full)

    def list(self):
        return np.unique(self.v)

    def clean(self):
        self.v = region_clean(region_shift(self.v
                                           , region_transform(self.v)))

    def split_label(self,l):
        label = (self.v == l)
        labels,num = ndimage.label(label > 0)
        if num < 1:
            return
        for idx,new_label in zip(
            range(1,num+1),
            range(l,l+num)):
            self.v[labels==idx] = new_label

    def edge_list():
        return edge_list(self.v)

    def copy(self):
        from copy import deepcopy
        cp = Label()
        cp.v = deepcopy(self.v)
        return cp
