# adj object
import numpy as np
import scipy
from scipy import ndimage

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

class Adj(object):
    def __init__(self,labels):
        self.v = adjacent(labels.v)
    
    def get_adj(self,label_list):
        """return all labels that are adjacent to label_list labels"""
        output = []+label_list  # include original labels as well
        # if self.v.shape[0] < max(label_list):    # trying to index new label so refresh adj
        #     self.v = adjacent(self.labels)
        for l in label_list:
            output += [i for i,x in enumerate(self.v[l,:]) if x]
        return set(output)

    # update
    def get_adj_radius(self,p,labels):
        import gui
        output = set()
        for i,v in np.ndenumerate(labels):
            if gui.dist(i,p[0:2]) < p[2]:
                output.add(labels[i])
        return output

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
