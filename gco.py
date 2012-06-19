# gco module
import sys,os,cv,cv2
import numpy as np
sys.path.insert(0,os.getcwd() + '/gcoc')
import gcoc
import scipy
from scipy import ndimage
from copy import deepcopy
import data

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

def region_clean(regions):
    out = np.zeros(regions.shape,dtype=regions.dtype)
    for l in range(regions.max()+1) :
        layer = (regions==l)
        if layer.any() :
            # labeled = ndimage.label(layer>0)[0]
            labeled = largest_connected_component(layer>0)
            # copy only the largest connected component
            out[np.nonzero(labeled == label_max(labeled))] = l
    # todo: what's happening with the zeros?
    return out

def smoothFn(s1,s2,l1,l2,adj):
    """smoothness function that could be passed to the minimzation"""
    if l1==l2 :
        return 0
    if not adj :
        return 10000000
    return int(1.0/(max(float(s1),float(s2))+1.0) * 255.0)
    # return int(1.0/float((abs(float(s1)-float(s2)) if \
        # abs(float(s1)-float(s2)) > 9 else 9)+1)*255.0)

def fit_region_z(im):
    """similar to fit_region"""
    mask_win = np.argwhere(im)
    (y0,x0),(y1,x1) = mask_win.min(0),mask_win.max(0)
    return (x0,y0,x1,y1)

def small_filter(labels,label_num):
    """cover small label region with the nearest label"""
    mask = create_mask(labels,[label_num]);
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

class Slice(object):
    def __init__(self, img, labels, shifted={}, win=(0,0), mask=None):
        """initialize fields and compute defaults"""
        # These values are created
        # when the class is instantiated.
        self.img = img.copy()
        self.labels = region_clean(region_shift(labels,
                                                region_transform(labels)))
        self.orig_labels = self.labels.copy()
        # self.num_labels = self.labels.max()+1
        self.data = data.Data(self.labels)
        # self.orig = np.array(self.data)
        self.orig = data.Data(self.labels)
        self.adj = adjacent(self.labels)
        self.shifted=shifted
        self.win=win
        self.shifted=shifted
        self.mask=mask

    def init_no_clean(self, img, labels, shifted={}, win=(0,0), mask=None):
        """initialize fields and compute defaults"""
        # These values are created
        # when the class is instantiated.
        self.img = img.copy()
        self.labels = labels
        self.orig_labels = self.labels.copy()
        # self.num_labels = self.labels.max()+1
        self.data = data.Data(self.labels)
        # self.orig = np.array(self.data)
        self.orig = data.Data(self.labels)
        self.adj = adjacent(self.labels)
        self.shifted=shifted
        self.win=win
        self.shifted=shifted
        self.mask=mask

    def get_adj(self,label_list):
        """return all labels that are adjacent to label_list labels"""
        output = []+label_list  # include original labels as well
        # if self.adj.shape[0] < max(label_list):    # trying to index new label so refresh adj
        #     self.adj = adjacent(self.labels)
        for l in label_list:
            output += [i for i,x in enumerate(self.adj[l,:]) if x]
        return set(output)

    def get_adj_radius(self,p):
        import gui
        output = set()
        for i,v in np.ndenumerate(self.labels):
            if gui.dist(i,p[0:2]) < p[2]:
                output.add(self.labels[i])
        return output

    def set_adj_new(self,adj=[]):
        """ add new row and column to self.adj for new label"""
        self.adj = np.hstack((np.vstack((self.adj,
                                         np.zeros((1,self.adj.shape[1])))),
                              np.zeros((self.adj.shape[0]+1,1))))
        self.adj[-1,-1] = True
        for a in adj:
            self.adj[-1,a] = True
            self.adj[a,-1] = True

    def set_adj_all(self):
        """remove all topology constraints (everything adjacent)"""
        self.adj[:,:] = True

    def set_adj_label_all(self,l):
        """remove topology constraints for a single label"""
        self.adj[l,:] = True
        self.adj[:,l] = True

    def edit_labels_gui(self,d):
        import gui
        while True:
            print("Starting GUI")
            w=gui.Window(self.img,self.labels)
            create=w.create_list
            remove=w.remove_list
            new_volumes = []
            for r in remove:
                # determine best dilation to kill region--might backfire
                (x0,y0,x1,y1) = fit_region(create_mask(self.labels,[r]))
                new_volumes.append(self.remove_label(r,max(x1-x0,y1-y0)+5))
            for c in create:
                new_volumes.append(self.add_label(c,d))
            for v in new_volumes:
                self.merge(v)
            if not create and not remove:
                break

    def edit_labels(self,d,addition,removal):
        create = [(b,a,5) for a,b in addition]
        remove = set([self.labels[(b,a)] for a,b in removal])
        new_volumes = []
        for r in remove:
            (x0,y0,x1,y1) = fit_region(create_mask(self.labels,[r]))
            new_volumes.append(self.remove_label(r,max(x1-x0,y1-y0)+5))
        for c in create:
            new_volumes.append(self.add_label(c,d))
        for v in new_volumes:
            self.merge(v)

    def remove_label(self,l,d):
        v = self.crop([l])
        v.data.dilate_all(d)
        v.data.label_inexclusive(v.labels,l)
        v.set_adj_all()
        v.graph_cut_no_clean(1)
        return v

    def add_label(self,p,d):
        v = self.crop(list(self.get_adj_radius(p)))
        p = (p[0]-v.win[0],p[1]-v.win[1],p[2])
        l = v.add_label_circle(p)
        v.data.dilate_label(l,p[2]*d)
        v.data.label_exclusive(v.labels,l)
        # v.output_data_term()
        v.set_adj_label_all(l)
        v.graph_cut_no_clean(1)
        return v

    def add_label_circle(self,p):
        import gui
        new_label = self.labels.max()+1
        for i,v in np.ndenumerate(self.labels):
            if gui.dist(i,p[0:2]) < p[2]:
                self.labels[i] = new_label
        # reconstruct data term after adding label
        # todo: do we really need to do these two lines?
        self.data = data.Data(self.labels) # ?
        self.adj = adjacent(self.labels)   # ?
        self.init_no_clean(self.img,
                           self.labels,
                           self.shifted,
                           self.win,
                           self.mask)
        return new_label

    def crop(self,label_list):
        """fork off subwindow volume"""
        label_list = self.get_adj(label_list)
        # import code; code.interact(local=locals())
        mask = create_mask(self.labels,label_list)
        (x0,y0,x1,y1) = fit_region(mask)
        # crop out everything with the given window
        mask_cropped = mask[y0:y1, x0:x1]
        cropped_seed = self.labels[y0:y1, x0:x1]
        new_img = self.img[y0:y1, x0:x1]
        # transform seed img
        # new_seed = np.zeros(cropped_seed.shape).astype('int16')
        new_seed = np.zeros_like(cropped_seed)
        label_transform = {}
        new_label = 1
        new_seed[np.logical_not(mask_cropped)] = 0
        for l in label_list:
            label_transform[new_label]=l
            # label_transform.append((l,new_label))
            new_seed[cropped_seed==l] = new_label
            new_label += 1
        return Slice(new_img,new_seed,label_transform,(y0,x0),mask_cropped)

    def merge(self,v):
        """merge another subwindow volume into this volume"""
        u = self.labels[v.win[0]:v.win[0]+v.labels.shape[0],
                        v.win[1]:v.win[1]+v.labels.shape[1]] # view into array
        new_label = self.labels.max()+1
        # herein lies the bug: a label might be eliminated
        old_labels = set(np.unique(v.labels))
        # bug propagates here
        shifted_labels = [v.shifted[x] 
                          for x in old_labels 
                          if x and x in v.shifted]

        # bug propagates here
        for l in [x for x in old_labels if x>0]:
            if l in v.shifted:
                # print("Shifting "+str(l)+" to "+str(v.shifted[l]))
                u[np.logical_and(v.labels==l,v.mask)]=v.shifted[l]
            else:
                # print("Shifting "+str(l)+" to "+str(new_label))
                u[v.labels==l]=new_label
                self.set_adj_new(shifted_labels)
                new_label+=1

        # self.__init__(self.img,
        #               region_clean(region_shift(self.labels,
        #                                         region_transform(self.labels))),
        #               self.shifted,
        #               self.win)

    def graph_cut(self,mode=0):
        """run graph cut on this volume (mode specifies V(p,q) term"""
        # if self.win != (0,0):
        # self.output_data_term()
        # w=gui.Window(self.img,self.labels)

        if not data.check_data_term(self.data.regions):
            print("Not all pixels have a label")
        else:
            print("All pixels have a label")

        output = gcoc.graph_cut(self.data.matrix(),
                                self.img,
                                self.labels,
                                self.adj,
                                self.labels.max()+1, # todo: extract from data
                                mode)
        self.__init__(self.img,
                      region_clean(region_shift(output,
                                                region_transform(output))),
                      self.shifted,
                      self.win)
        # self.labels = region_clean(region_shift(output,
        #                                         region_transform(output)))
        return self.labels

    def graph_cut_no_clean(self,mode=0):
        """run graph cut on this volume (mode specifies V(p,q) term"""
        output = gcoc.graph_cut(self.data.matrix(),
                                self.img,
                                self.labels,
                                self.adj,
                                self.labels.max()+1, # todo: extract from data
                                mode)
        self.labels = output.copy()
        return self.labels
