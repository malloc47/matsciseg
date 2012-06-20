# adj object
import numpy as np
import scipy
from scipy import ndimage
import gcoc

def adjacent(labels):
    """determine which labels are adjacent"""
    adj = np.zeros((labels.max()+1,labels.max()+1),dtype=bool)
    def set_adj(i,j):
        adj[i,j] = True
        adj[j,i] = True

    # list of quad tuples that define the i,j ranges and the i,j transforms
    ranges = [ (range(0,labels.shape[0]-1),
                range(0,labels.shape[1]),
                lambda i:i+1, lambda j:j),

               (range(0,labels.shape[0]),
                range(0,labels.shape[1]-1),
                lambda i:i, lambda j:j+1),

               (range(1,labels.shape[0]),
                range(0,labels.shape[1]),
                lambda i:i-1, lambda j:j),

               (range(0,labels.shape[0]),
                range(1,labels.shape[1]),
                lambda i:i, lambda j:j-1),

               (range(0,labels.shape[0]-2), # don't know why these were included
                range(0,labels.shape[1]),
                lambda i:i+2, lambda j:j),

               (range(0,labels.shape[0]),
                range(0,labels.shape[1]-2),
                lambda i:i, lambda j:j+2) ]

    for rg in ranges:
        for i in rg[0]:
            for j in rg[1]:
                r = labels[i,j]
                l = labels[rg[2](i),rg[3](j)]
                if r != l : set_adj(r,l)

    return adj

def circle(p,shape):
    output = np.zeros(shape,dtype='bool')
    r = p[2]
    cx, cy = p[0:2]
    y, x = np.ogrid[-r:r+1, -r:r+1]
    index = x**2 + y**2 <= r**2
    # handle border cases
    crop = ((cx-r)-max(cx-r,0), 
            min(cx+r+1,output.shape[0])-(cx+r+1),
            (cy-r)-max(cy-r,0),
            min(cy+r+1,output.shape[1])-(cy+r+1))
    output[cx-r-crop[0]:cx+r+crop[1]+1, cy-r-crop[2]:cy+r+crop[3]+1][
        index[abs(crop[0]):index.shape[0]-abs(crop[1]),
              abs(crop[2]):index.shape[1]-abs(crop[3])]] = True
    return output

# def circle_mask(r,shape,p):
#     y,x = np.ogrid[-shape[0]:shape[0], -shape[1]:shape[1]]
#     return (x-(p[0]-shape[0]*2))**2+(y-(p[1]-shape[1]*2))**2 <= r**2

class Adj(object):
    def __init__(self,labels):
        self.v = gcoc.adjacent(labels.v,labels.max()+1)
        # self.v = adjacent(labels.v)
    
    def get_adj(self,label_list):
        """return all labels that are adjacent to label_list labels"""
        output = []+label_list  # include original labels as well
        # if self.v.shape[0] < max(label_list):    # trying to index new label so refresh adj
        #     self.v = adjacent(self.labels)
        for l in label_list:
            output += [i for i,x in enumerate(self.v[l,:]) if x]
        return set(output)

    # doesn't really belong in adj, per-se
    def get_adj_radius(self,p,labels):
        """get all regions covered by circle"""
        return set(np.unique(labels[
                    circle(p,labels.shape)]))

    def set_adj_new(self,adj=[]):
        """ add new row and column to self.v for new label"""
        self.v = np.hstack((np.vstack((self.v,
                                         np.zeros((1,self.v.shape[1])))),
                              np.zeros((self.v.shape[0]+1,1))))
        self.v[-1,-1] = True
        for a in adj:
            self.v[-1,a] = True
            self.v[a,-1] = True

    def set_adj_all(self):
        """remove all topology constraints (everything adjacent)"""
        self.v[:,:] = True

    def set_adj_label_all(self,l):
        """remove topology constraints for a single label"""
        self.v[l,:] = True
        self.v[:,l] = True
