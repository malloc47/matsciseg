# gco module
import sys,os,cv,cv2
import numpy as np
import gcoc
import scipy
from scipy import ndimage
from copy import deepcopy
import data, adj, label
import operator
import math

def smoothFn(s1,s2,l1,l2,adj):
    """smoothness function that could be passed to the minimzation"""
    if l1==l2 :
        return 0
    if not adj :
        return 10000000
    return int(1.0/(max(float(s1),float(s2))+1.0) * 255.0)
    # return int(1.0/float((abs(float(s1)-float(s2)) if \
        # abs(float(s1)-float(s2)) > 9 else 9)+1)*255.0)

def crop_box(a,box):
    (x0,y0,x1,y1) = box
    return a[y0:y1, x0:x1]

def candidate_point(p,q,r):
    new_q = tuple([ j-i for i,j in zip(p,q)])
    theta = math.atan2(*new_q[::-1])
    if theta < 0:
        theta += 2*math.pi
    print(str(theta))
    return (int(r*math.cos(theta))+p[0],
            int(r*math.sin(theta))+p[1])

class Slice(object):
    def __init__(self, img, labels, shifted={}, 
                 win=(0,0), mask=None, lightweight=False,
                 nodata=False):
        """initialize fields and compute defaults"""
        # These values are created when the class is instantiated.
        self.img = img.copy()
        if lightweight:
            self.labels = label.Label()
            self.labels.v = labels
        else:
            self.labels = label.Label(labels)
        # self.orig_labels = self.labels.copy()
        # self.num_labels = self.labels.max()+1
        if nodata:
            self.data = None
        else:
            self.data = data.Data(self.labels.v)
            self.orig = self.data.copy() # potential slowdown
        self.adj = adj.Adj(self.labels) # adjacent(self.labels)
        self.shifted=shifted
        self.win=win
        self.mask=mask

    def edit_labels_gui(self,d):
        import gui
        while True:
            print("Starting GUI")
            w=gui.Window(self.img,self.labels.v)
            create = [(a,b,c,d) for a,b,c in w.create_list]
            remove=w.remove_list
            self.process_annotations(create,remove)
            if not create and not remove:
                break

    def edit_labels(self,addition=[],removal=[],line=[]):
        removal = set([self.labels.v[(a,b)] for a,b in removal])
        self.process_annotations(addition,removal,line)

    def process_annotations(self,create=[],remove=[],line=[]):
        new_volumes = []
        for r in remove:
            (x0,y0,x1,y1) = label.fit_region(self.labels.create_mask([r]))
            new_volumes.append(self.remove_label(r,max(x1-x0,y1-y0)+5))
        for c in create:
            new_volumes.append(self.add_label_circle(c))
        for l in line:
            new_volumes.append(self.add_label_line(l))
        for v in new_volumes:
            self.merge(v)
        self.__init__(self.img
                      , self.labels.v
                      , self.shifted
                      , self.win
                      , self.mask
                      , lightweight=True
                      , nodata=True)

# lut = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,1,1,0,0]
    def clique_swap(self,d):
        for l in range(0,self.labels.max()):
            v = self.crop([l])
            if v is None:
                continue
            # v.data.dilate(d)
            v.data.dilate_fixed_center(d, rel_size=0.1, min_size=15)
            v.data.label_exclusive(0, v.labels.v == 0)
            v.data.pixels_exclusive(
                v.labels.region_outline()
                # + [(i,j,l) for (i,j,l) in v.labels.centers_of_mass() if l > 0]
                )
            v.graph_cut(1,lite=False)
            self.merge(v)
            
    def remove_label_dilation(self,l,d):
        """removal that uses a large dilation to (hopefully) remove
        the candidate region"""
        v = self.crop([l])
        v.data.dilate_all(d)
        shifted = v.rev_shift(l)
        v.data.regions[shifted] = v.labels.v==shifted
        v.data.label_inexclusive(v.labels.v==shifted,shifted)
        v.adj.set_adj_all()
        v.graph_cut(1,lite=True)
        return v

    def remove_label_old(self,l,d):
        """removal that truly removes region"""
        v = self.crop([l])
        v.data.or_all(crop_box(data.select_region(self.labels.v,l),
                               (v.win[1]
                                , v.win[0]
                                , v.win[1] + v.labels.v.shape[1]
                                , v.win[0] + v.labels.v.shape[0])),
                      skip=[0])
        new_l = (key for key,value in v.shifted.items() if value==l).next()
        v.data.label_erase(new_l)
        v.adj.set_adj_all()
        v.graph_cut(1)
        return v

    def remove_label(self,l,d):
        """removal that truly removes region"""
        v = self.crop([l],[l])
        new_mask = crop_box(data.select_region(self.labels.v,l),
                               (v.win[1]
                                , v.win[0]
                                , v.win[1] + v.labels.v.shape[1]
                                , v.win[0] + v.labels.v.shape[0]))
        v.data.or_term(new_mask)
        v.adj.set_adj_all()
        v.graph_cut(1)
        return v

    def add_label_circle(self,p):
        v = self.crop(list(self.adj.get_adj_radius(p,self.labels.v)))
        p = (p[0]-v.win[0],p[1]-v.win[1],p[2],p[3])
        l = v.new_label_circle(p)
        v.data.label_exclusive(l,v.labels.v==l)
        # v.data.dilate_label(l,p[3])
        # directly set data term instead of dilating--matches gui
        v.data.regions[l] = adj.circle((p[0],p[1],p[2]+p[3]),v.labels.v.shape)
        v.data.label_exclusive(0,v.labels.v==0)
        v.adj.set_adj_label_all(l)
        v.graph_cut(1)
        return v

    def add_label_line(self,p):
        line_img = data.line(p,self.labels.v.shape,p[4])
        v = self.crop(list(np.unique(self.labels.v[line_img])))
        p = (p[0]-v.win[0],p[1]-v.win[1],p[2]-v.win[0],p[3]-v.win[1],p[4],p[5])
        l = v.new_label_line(p)
        v.data.label_exclusive(l,v.labels.v==l)
        v.data.regions[l] = data.line(p,v.labels.v.shape,p[4]+p[5])
        v.data.label_exclusive(0,v.labels.v==0)
        v.adj.set_adj_label_all(l)
        v.graph_cut(1)
        return v

    def new_label_circle(self,p):
        new_label = self.labels.max()+1
        self.labels.v[adj.circle(p,self.labels.v.shape)] = new_label
        # reconstruct data term after adding label
        # todo: constructing these before this is useless work
        self.data = data.Data(self.labels.v)
        self.adj = adj.Adj(self.labels)
        return new_label

    def new_label_line(self,p):
        new_label = self.labels.max()+1
        self.labels.v[data.line(p,self.labels.v.shape,p[4])] = new_label
        self.data = data.Data(self.labels.v)
        self.adj = adj.Adj(self.labels)
        return new_label

    def non_homeomorphic_remove(self,d,size):
        s = self.labels.sizes()
        print(str([l for l in s if l < size]))
        for l in [ l for (l,s) in zip(range(len(s)),s) if s < size ]:
            v = self.remove_label_dilation(l,d)
            self.merge(v)

    def non_homeomorphic_yjunction(self,d,r=3,r2=4,corr=0.66,min_size=100):
        import pylab
        import matplotlib.cm as cm
        import matsci.gui
        t = np.mean(self.img)*0.75
        j = self.labels.junctions(r)
        sizes = self.labels.sizes()
        for (p,ls) in j:
            print(str(p))
            print(str(ls))
            if min([sizes[s] for s in ls]) < min_size:
                continue
            v = self.crop(list(ls),extended=False)
            pylab.imshow(
                matsci.gui.color_jet(
                    matsci.gui.grey_to_rgb(v.img)
                    , v.labels.v))
            pylab.hold(True)
            p_shifted = tuple([ j-i for i,j in zip(v.win,p)])
            pylab.plot([p_shifted[1]],[p_shifted[0]],'r.',markersize=d)
            candidates = [ (x,y,l) for ((x,y),l) in v.yjunction_candidates(p,r2) ]
            print(str(candidates))
            for c in candidates:
                pylab.plot([c[1]],[c[0]],'g.',markersize=d)
            for c in v.labels.centers_of_mass():
                pylab.plot([c[1]],[c[0]],'b.',markersize=d)
            pylab.show()
            cuts = []
            for c in candidates:
                v2 = v.copy()
                v2.new_label_circle(c)
                v2.data.dilate(d)
                v2.data.pixels_exclusive([(i,j,l) for (i,j,l) in 
                                          v2.labels.centers_of_mass() if l > 0])
                v2.graph_cut(1)
                score = v2.labels.region_boundary_intensity(v2.img,v2.labels.max(),t)
                cuts += [(v2,score)]
            best = max(cuts,key=operator.itemgetter(1))
            if best[1] > corr:
                print('#$%L@#KJ$LSEKRJSZ I GIOT A HIGET')
                self.merge(best[0])
            
    def yjunction_candidates(self,p,r):
        p_shifted = tuple([ j-i for i,j in zip(self.win,p)])
        return [ (candidate_point(p_shifted,(x,y),r),l)
                 for x,y,l in self.labels.centers_of_mass() if l > 0 ]

    def rev_shift(self,l):
        return dict((v,k) for k,v in self.shifted.items())[l]

    def crop(self,label_list,blank=[],extended=True):
        """fork off subwindow volume"""
        if extended:
            label_list = self.adj.get_adj(label_list)
        if len(label_list) < 2:
            return None
        mask = self.labels.create_mask(label_list)
        box = label.fit_region(mask)
        # crop out everything with the given window
        mask_cropped = crop_box(mask,box)
        cropped_seed = crop_box(self.labels.v,box)
        new_img      = crop_box(self.img,box)
        # transform seed img
        new_seed = np.zeros_like(cropped_seed)
        label_transform = {}
        new_label = 1
        for l in label_list:
            label_transform[new_label]=l
            # print("Shifting "+str(l)+" to "+str(new_label))
            if not l in blank:
                new_seed[cropped_seed==l] = new_label
                new_label += 1
            else:
                new_seed[cropped_seed==l] = 0
        # new_seed[np.logical_not(mask_cropped)] = 0
        return Slice(new_img
                     , new_seed
                     , label_transform
                     , (box[1],box[0])
                     , mask_cropped
                     , lightweight=True)

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

    def graph_cut(self,mode=0,lite=False):
        """run graph cut on this volume (mode specifies V(p,q) term"""
        # if self.win != (0,0):
        # self.output_data_term()
        # w=gui.Window(self.img,self.labels)

        # if not data.check_data_term(self.data.regions):
        #     print("Not all pixels have a label")
        # else:
        #     print("All pixels have a label")

        # print("region size: " + str(self.img.shape))
        # print("min: " + str(self.labels.v.min()))
        # print("max: " + str(self.labels.v.max()))

        output = gcoc.graph_cut(self.data.matrix(),
                                self.img,
                                np.array(self.labels.v).astype('int16'),
                                # self.labels.v,
                                self.adj.v,
                                self.labels.max()+1, # todo: extract from data
                                mode)
        # fully reinitialize self (recompute adj, data, etc.)
        self.__init__(self.img
                      , output
                      , self.shifted
                      , self.win
                      , self.mask
                      , lightweight=lite
                      , nodata=lite)
        return self.labels.v

    def copy(self):
        from copy import deepcopy
        cp = Slice(self.img
                   , self.labels.v.copy()
                   , self.shifted
                   , self.win
                   , self.mask
                   , lightweight=True
                   , nodata=True)
        cp.data = self.data.copy()
        return cp
