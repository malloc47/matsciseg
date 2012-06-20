# gco module
import sys,os,cv,cv2
import numpy as np
sys.path.insert(0,os.getcwd() + '/gcoc')
import gcoc
import scipy
from scipy import ndimage
from copy import deepcopy
import data, adj, label

def smoothFn(s1,s2,l1,l2,adj):
    """smoothness function that could be passed to the minimzation"""
    if l1==l2 :
        return 0
    if not adj :
        return 10000000
    return int(1.0/(max(float(s1),float(s2))+1.0) * 255.0)
    # return int(1.0/float((abs(float(s1)-float(s2)) if \
        # abs(float(s1)-float(s2)) > 9 else 9)+1)*255.0)

class Slice(object):
    def __init__(self, img, labels, shifted={}, win=(0,0), mask=None):
        """initialize fields and compute defaults"""
        # These values are created
        # when the class is instantiated.
        self.img = img.copy()
        self.labels = label.Label(labels)
        # self.orig_labels = self.labels.copy()
        # self.num_labels = self.labels.max()+1
        self.data = data.Data(self.labels.v)
        # self.orig = np.array(self.data)
        self.orig = data.Data(self.labels.v) # potential slowdown
        self.adj = adj.Adj(self.labels) # adjacent(self.labels)
        self.shifted=shifted
        self.win=win
        self.shifted=shifted
        self.mask=mask

    def init_no_clean(self, img, labels, shifted={}, win=(0,0), mask=None):
        """initialize fields and compute defaults"""
        # These values are created
        # when the class is instantiated.
        self.img = img.copy()
        # self.labels = labels
        self.labels = label.Label(labels)
        # self.orig_labels = self.labels.copy()
        # self.num_labels = self.labels.max()+1
        self.data = data.Data(self.labels.v)
        # self.orig = np.array(self.data)
        self.orig = data.Data(self.labels.v) # potential slowdown
        self.adj = adj.Adj(self.labels)
        self.shifted=shifted
        self.win=win
        self.shifted=shifted
        self.mask=mask


    def edit_labels_gui(self,d):
        import gui
        while True:
            print("Starting GUI")
            w=gui.Window(self.img,self.labels.v)
            create=w.create_list
            remove=w.remove_list
            new_volumes = []
            for r in remove:
                # determine best dilation to kill region--might backfire
                (x0,y0,x1,y1) = label.fit_region(self.labels.create_mask([r]))
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
            (x0,y0,x1,y1) = label.fit_region(self.labels.create_mask([r]))
            new_volumes.append(self.remove_label(r,max(x1-x0,y1-y0)+5))
        for c in create:
            new_volumes.append(self.add_label(c,d))
        for v in new_volumes:
            self.merge(v)

    def remove_label(self,l,d):
        v = self.crop([l])
        v.data.dilate_all(d)
        v.data.label_inexclusive(v.labels.v,l)
        v.adj.set_adj_all()
        v.graph_cut(1)
        return v

    # from profilehooks import profile

    # @profile
    def add_label(self,p,d):
        v = self.crop(list(self.adj.get_adj_radius(p,self.labels.v)))
        p = (p[0]-v.win[0],p[1]-v.win[1],p[2])
        l = v.add_label_circle(p)
        v.data.dilate_label(l,p[2]*d)
        v.data.label_exclusive(v.labels.v,l)
        # v.output_data_term()
        v.adj.set_adj_label_all(l)
        v.graph_cut(1)
        return v

    def add_label_circle(self,p):
        import gui
        new_label = self.labels.max()+1
        self.labels.v[adj.circle(p,self.labels.v.shape)] = new_label
        # reconstruct data term after adding label
        # todo: do we really need to do these two lines?
        self.data = data.Data(self.labels.v)
        self.adj = adj.Adj(self.labels)
        return new_label

    def crop(self,label_list):
        """fork off subwindow volume"""
        label_list = self.adj.get_adj(label_list)
        # import code; code.interact(local=locals())
        mask = self.labels.create_mask(label_list)
        (x0,y0,x1,y1) = label.fit_region(mask)
        # crop out everything with the given window
        mask_cropped = mask[y0:y1, x0:x1]
        cropped_seed = self.labels.v[y0:y1, x0:x1]
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
        u = self.labels.v[v.win[0]:v.win[0]+v.labels.v.shape[0],
                        v.win[1]:v.win[1]+v.labels.v.shape[1]] # view into array
        new_label = self.labels.max()+1
        # herein lies the bug: a label might be eliminated
        old_labels = set(np.unique(v.labels.v))
        # bug propagates here
        shifted_labels = [v.shifted[x] 
                          for x in old_labels 
                          if x and x in v.shifted]

        # bug propagates here
        for l in [x for x in old_labels if x>0]:
            if l in v.shifted:
                # print("Shifting "+str(l)+" to "+str(v.shifted[l]))
                u[np.logical_and(v.labels.v==l,v.mask)]=v.shifted[l]
            else:
                # print("Shifting "+str(l)+" to "+str(new_label))
                u[v.labels.v==l]=new_label
                # self.adj.set_adj_new(shifted_labels)
                new_label+=1
                self.__init__(self.img, 
                      self.labels.v, 
                      self.shifted, 
                      self.win, 
                      self.mask)

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

        print("region size: " + str(self.img.shape))
        print("min: " + str(self.labels.v.min()))
        print("max: " + str(self.labels.v.max()))

        output = gcoc.graph_cut(self.data.matrix(),
                                self.img,
                                self.labels.v,
                                self.adj.v,
                                self.labels.max()+1, # todo: extract from data
                                mode)
        # fully reinitialize self (recompute adj, data, etc.)
        self.__init__(self.img, 
                      output, 
                      self.shifted, 
                      self.win, 
                      self.mask)
        return self.labels.v

    def graph_cut_no_clean(self,mode=0):
        """run graph cut on this volume (mode specifies V(p,q) term"""
        output = gcoc.graph_cut(self.data.matrix(),
                                self.img,
                                self.labels.v,
                                self.adj.v,
                                self.labels.max()+1, # todo: extract from data
                                mode)
        # self.labels = output.copy()
        return output
