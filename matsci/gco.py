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

def smoothFnSigma(s1,s2,l1,l2,adj,sigma):
    """smoothness function that could be passed to the minimzation"""
    if l1==l2 :
        return 0
    if not adj :
        return 10000000
    return int(math.exp( -1 * (((float(s1)-float(s2))**2 )/(2*(float(sigma)**2)))) * 255.0)

def smoothFnSigmaMax(s1,s2,l1,l2,adj,sigma):
    """smoothness function that could be passed to the minimzation"""
    if l1==l2 :
        return 0
    if not adj :
        return 10000000
    return int(math.exp( -1 * ((max(float(s1),float(s2))**2 )/(2*(float(sigma)**2)))) * 255.0)

def crop_box(a,box):
    (x0,y0,x1,y1) = box
    return a[y0:y1, x0:x1]

def candidate_point(p,q,r):
    new_q = tuple([ j-i for i,j in zip(p,q)])
    theta = math.atan2(*new_q[::-1])
    if theta < 0:
        theta += 2*math.pi
    return (int(r*math.cos(theta))+p[0],
            int(r*math.sin(theta))+p[1])

def estimate_size(c,labels):
    max_size = 50
    lower_size = 2
    fg = labels ==labels[c[0],c[1]]
    d = 1
    while True :
        circle = adj.circle((c[0],c[1],d),labels.shape)
        if np.sum(circle) > np.sum(np.logical_and(circle,fg)) or d >= max_size :
            break
        d = d+1
    if d > lower_size+1:
        return d-lower_size
    else:
        return d if d > lower_size-1 else lower_size
    

class Slice(object):
    def __init__(self, img, labels, shifted={}, 
                 win=(0,0), mask=None, lightweight=False,
                 nodata=False, center=None, bg=False, adjin=None):
        """initialize fields and compute defaults"""
        # These values are created when the class is instantiated.
        self.img = img.copy()
        if lightweight:
            self.labels = label.Label()
            self.labels.v = labels
        else:
            self.labels = label.Label(
                labels, 
                boundary=(None if center is None else labels==0))
        # self.orig_labels = self.labels.copy()
        # self.num_labels = self.labels.max()+1
        if nodata:
            self.data = data.Data()
        else:
            self.data = data.Data(self.labels.v)
            # self.orig = self.data.copy() # potential slowdown
        if adjin is None:
            self.adj = adj.Adj(self.labels) # adjacent(self.labels)
        else:
            self.adj = adj.Adj()
            self.adj.v = adjin
        self.shifted=shifted
        self.win=win
        self.mask=mask
        self.center=center
        self.bg=bg

    @classmethod
    def load(cls,in_file):
        npz = np.load(in_file)
        c = cls(npz['img'], npz['label'], nodata=True, lightweight=True, adjin=npz['adj'])
        c.data.regions = data.unstack_matrix(npz['data'])
        return c

    def save(self,out):
        img,label,data,adj = self.dump()
        np.savez(out,img=img,label=label,data=data,adj=adj)

    def dump(self):
        return (self.img,self.labels.v,data.stack_matrix(self.data.regions),self.adj.v)

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

    def edit_labels(self,addition=[],auto=[],removal=[],line=[]):
        removal = set([self.labels.v[(a,b)] for a,b in removal])
        self.process_annotations(create=addition,auto=auto,remove=removal,line=line)

    def process_annotations(self,create=[],auto=[],remove=[],line=[]):
        new_volumes = []
        print(str(auto))
        for r in remove:
            (x0,y0,x1,y1) = label.fit_region(self.labels.create_mask([r]))
            new_volumes.append(self.remove_label(r,max(x1-x0,y1-y0)+5))
        for c in create:
            new_volumes.append(self.add_label_circle(c))
        for t in auto:
            new_volumes.append(self.add_label_circle_auto(t))
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
                      , nodata=True
                      , center=getattr(self,'center',None))

# lut = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,1,1,0,0]
    def clique_swap(self,d,f=None):
        for l in range(0,self.labels.max()):
            v = self.crop([l])
            if v is None:
                continue
            if not f is None and f(v):
                print('skipping ' + str(v.center))
                continue
            # v.data.dilate(d)
            if(d>0):
                v.data.dilate_fixed_center(d, rel_size=0.1, min_size=15)
            v.data.label_exclusive(0, v.labels.v == 0)
            v.data.pixels_exclusive(
                v.labels.region_outline()
                # + [(i,j,l) for (i,j,l) in v.labels.centers_of_mass() if l > 0]
                )
            v.graph_cut(1,lite=False)
            self.merge(v)
        return self.labels.v

    def local_adj(self):
        return [ self.crop([l]).adj for l in range(0,self.labels.max()) ]

    def local(self):
        return [ (self.crop([l]),l) for l in range(0,self.labels.max()) ]

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

    def add_label_circle(self,p,crop=True):
        if crop:
            v = self.crop(list(self.adj.get_adj_radius(p,self.labels.v)))
            p = (p[0]-v.win[0],p[1]-v.win[1],p[2],p[3])
        else:
            v = self
        l = v.new_label_circle(p)
        v.data.label_exclusive(l,v.labels.v==l)
        # v.data.dilate_label(l,p[3])
        # directly set data term instead of dilating--matches gui
        v.data.regions[l] = adj.circle((p[0],p[1],p[2]+p[3]),v.labels.v.shape)
        v.data.label_exclusive(0,v.labels.v==0)
        v.adj.set_adj_label_all(l)
        v.graph_cut(1)
        return v

    def add_label_circle_auto(self,p,crop=True):
        if crop:
            v = self.crop(list(self.adj.get_adj_radius(p,self.labels.v)))
            p = (p[0]-v.win[0],p[1]-v.win[1],p[2],p[3])
        else:
            v = self
        d = estimate_size(p,v.labels.v)
        print('auto d : '+str(d))
        p = (p[0],p[1],d,2*d)
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

    def new_dummy_label(self):
        new_label = self.labels.max()+1
        self.data.add_dummy_label()
        self.adj.set_adj_new()
        self.adj.set_adj_all()
        # self.adj.set_adj_label_all(new_label)
        return new_label

    def non_homeomorphic_remove(self,d,size):
        s = self.labels.sizes()
        # print(str([l for l in s if l < size]))
        for l in [ l for (l,s) in zip(range(len(s)),s) if s < size ]:
            v = self.remove_label_dilation(l,d)
            self.merge(v)

    def non_homeomorphic_yjunction(self, d=20, r1=1, r2=1, r3=7, corr=0.66,
                                   min_size=100, new_min_size=25):
        """
        r1 = new point radius
        r2 = new point dilation
        r3 = distance from y-junction to place seed
        """
        import pylab
        import matplotlib.cm as cm
        import matsci.gui

        def imshow(img):
            pylab.clf()
            pylab.imshow(img,cmap=cm.Greys_r)
            pylab.show()

        t = np.mean(self.img)*1.5
        print('Image mean: '+str(np.mean(self.img)))
        j = self.labels.junctions(r1)
        sizes = self.labels.sizes()
        final = []
        for (p,ls) in j:
            if min([sizes[s] for s in ls]) < min_size:
                continue
            v = self.crop(list(ls),extended=False)

            # pylab.clf()
            # pylab.imshow(
            #     matsci.gui.color_jet(
            #         matsci.gui.grey_to_rgb(v.img)
            #         , v.labels.v))
            # pylab.hold(True)
            # p_shifted = tuple([ j-i for i,j in zip(v.win,p)])
            # pylab.plot([p_shifted[1]],[p_shifted[0]],'r.',markersize=d)

            candidates = [(x,y,l) for ((x,y),l) in v.yjunction_candidates(p,r3)]

            # print(str(candidates))
            # for c in candidates:
            #     pylab.plot([c[1]],[c[0]],'g.',markersize=d)
            # for c in [(i,j,l) for (i,j,l) in 
            #           v.labels.centers_of_mass() if l > 0]:
            #     pylab.plot([c[1]],[c[0]],'b.',markersize=d)

            # pylab.show()

            cuts = []
            for c in candidates:
                v2 = v.copy()
                new_l = v2.new_label_circle(c)
                v2.data.dilate(d)
                v2.data.pixels_exclusive([(i,j,l) for (i,j,l) in 
                                          v2.labels.centers_of_mass() 
                                          if l > 0 and v2.labels.v[i,j]==l])
                # v2.data.label_exclusive(new_l,adj.circle((c[0],c[1],r2),v2.labels.v.shape))
                # v2.data.regions[new_l] = adj.circle((c[0],c[1],d),v2.labels.v.shape)
                v2.data.label_exclusive(0,v2.labels.v==0)
                v2.adj.set_adj_label_all(new_l)

                # imshow(v2.data.output_data_term())

                v2.graph_cut(1)

                # new_l = v2.new_label_circle(c)
                # # v2.data.dilate_fixed_center(d, rel_size=0.1, min_size=15,first=False)
                # v2.data.dilate(d)
                # # v2.data.pixels_exclusive([(i,j,l) for (i,j,l) in 
                # #                           v2.labels.centers_of_mass() if l > 0])
                # # v2.data.dilate_fixed_center(d, rel_size=0.1, min_size=15,first=False)
                # imshow(v2.data.output_data_term())
                # v2.data.regions[new_l] = adj.circle((c[0],c[1],r1+r2),v2.labels.v.shape)
                # imshow(v2.data.output_data_term())
                # v2.data.label_exclusive(new_l,adj.circle((c[0],c[1],r1),v2.labels.v.shape))
                # imshow(v2.data.output_data_term())
                # v2.data.label_exclusive(0,v2.labels.v==0)
                # imshow(v2.data.output_data_term())
                # v2.adj.set_adj_label_all(new_l)
                # v2.data.label_exclusive(0,v2.labels.v==0)
                # v2.graph_cut(1,lite=True)
                # v2.add_label_circle(c+(r2,),crop=False)
                if(np.count_nonzero(v2.labels.v == new_l) > new_min_size):
                    # img2 = v2.img.copy()
                    # img2[np.logical_not(label.binary_remove(v2.labels.v == new_l))] = 0
                    # imshow(img2)
                    score = v2.labels.region_boundary_intensity(v2.img,new_l,t)
                    cuts += [(v2,score)]
            if cuts:
                best = max(cuts,key=operator.itemgetter(1))
                if best[1] > corr:
                    print('Creating new region: '+str(best[1]))
                    final += [best[0]]
        print('Number of new segments: '+str(len(final)))
        for f in final:
            self.merge(f)
            
    def yjunction_candidates(self,p,r):
        p_shifted = tuple([ j-i for i,j in zip(self.win,p)])
        return [ (candidate_point(p_shifted,(x,y),r),r)
                 for x,y,l in self.labels.centers_of_mass() if l > 0 ]

    def rev_shift(self,l):
        return dict((v,k) for k,v in self.shifted.items())[l]

    def crop(self, label_list, blank=[], extended=True, padding=0, no_bg=False):
        """fork off subwindow volume"""
        if extended:
            new_label_list = self.adj.get_adj(label_list)
        else:
            new_label_list = label_list
        # if len(new_label_list) < 2: # Regression warning: untested
        if len(new_label_list) < 1:
            return None
        mask = self.labels.create_mask(new_label_list)
        try:
            box = label.fit_region(mask, padding)
        except:
            return None
        # crop out everything with the given window
        mask_cropped = crop_box(mask,box)
        cropped_seed = crop_box(self.labels.v,box)
        new_img      = crop_box(self.img,box)
        # transform seed img
        new_seed = np.zeros_like(cropped_seed)
        label_transform = {}
        new_label = 1
        for l in new_label_list:
            label_transform[new_label]=l
            # print("Shifting "+str(l)+" to "+str(new_label))
            if not l in blank:
                new_seed[cropped_seed==l] = new_label
                new_label += 1
            else:
                new_seed[cropped_seed==l] = 0
        if no_bg:
            new_seed = label.small_filter(new_seed,0)
        # new_seed[np.logical_not(mask_cropped)] = 0
        return Slice(new_img
                     , new_seed
                     , label_transform
                     , (box[1],box[0])
                     , mask_cropped
                     , lightweight=True
                     , center = label_list[0] if len(label_list) < 2 else None)

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

    def alpha_expansion(self, dilation=10, mode=1, bias=1):
        for i in self.labels.list():
            print(str(i))
            v = self.crop([i], extended=False, padding=dilation+5)
            if v is None:
                print('Label '+str(i)+' is empty')
                continue
            if not ( (v.labels.v==0).any() or (v.labels.v==1).any() ):
                print('Label '+str(i)+' is empty')
                continue
            v.data.dilate_fixed_center(dilation, rel_size=0.1, min_size=2, 
                                       first=True,single=True)
            v.graph_cut(mode=mode, bias=bias, tc_iter=5)
            v.mask = np.ones_like(v.data.regions[0])
            self.merge(v)

    def alpha_beta_swap(self, dilation=10, mode=1, bias=1):
        """alpha-beta swap method, written in python and using existing
        crop/merge algorithm to carry it out in an efficient manner"""

        # def remove_bg(v):
        #     labels = label.small_filter(v.labels.v,0)
        #     return data.Data(v.labels.v)

        for i,j in [ (i,j) for i,j, in self.adj.pairs() if i>=0 and j>=0]:
            print(str((i,j)))
            v = self.crop([i,j], extended=False)
            v.adj.set_unadj_bg()
            v.data.dilate(dilation)
            v.data.convert_to_int16()
            v.graph_cut(mode=mode, bias=bias)
            # if max(label.num_components(output)) > 1:
            #     print('Rerunning')
            #     v = self.crop([i,j], extended=False, no_bg=True)
            #     v.adj.set_unadj_bg()
            #     v.data.dilate(dilation)
            #     v.data.convert_to_int16()
            #     v.graph_cut(mode=mode, bias=bias)
            self.merge(v)

    def graph_cut(self,mode=0,lite=False,bias=1,sigma=None,replace=None,tc_iter=0):
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

        # print(str(self.data.matrix().shape))
        # print(str(self.img.shape))
        # print(str(np.array(self.labels.v).shape))
        # print(str(self.adj.v.shape))

        if sigma is None:
            sigma = np.std(self.img) if mode > 2 else 10

        # assume int if not bool
        if replace is None:
            replace = 0 if (self.data.regions[0].dtype.kind == 'b') else -1

        output = gcoc.graph_cut(self.data.matrix()
                                , self.img
                                , np.array(self.labels.v)#.astype('int16')
                                # , self.labels.v,
                                , self.adj.v
                                # , self.labels.max()+1 # todo: extract from data
                                , self.data.length()
                                , mode
                                , sigma
                                , bias
                                , replace
                                )

        if tc_iter >= 1 and \
                len(self.data.regions) == 2 and \
                max(label.num_components(output,full=False)) > 1:

            sys.path.insert(0,'./pytopocut/')
            import topocut

            num_hole = label.num_components(output,full=False)[0] - 1
            num_comp = label.num_components(output,full=False)[1]
            num_iter = 0

            self.data.convert_to_x(data.bool_to_float)  # topocut requires float

            while ((num_comp > 1 or num_hole > 0) and num_iter < tc_iter):
                num_iter += 1
                # print('running topocut iteration '+str(num_iter))

                # first round takes care of components reliably
                # without encountering a degenerate case
                # (non-stocastic)
                ubg, ufg, phi, num_comp, num_hole = \
                    topocut.topofix.fix(self.data.regions[0].astype('float64')
                                        , self.data.regions[1].astype('float64')
                                        , output,1,-1, stocastic=False)

                # # not 100% sure about this escape clause...
                # if (num_comp <= 1) and ( num_hole <= 0):
                #     output = (phi < 0)
                #     break

                # convert back to ints
                self.data.regions[0] = ubg.astype('int16')
                self.data.regions[1] = ufg.astype('int16')

                new_data = self.data.matrix()
                # make "infinity" value unused by setting it to max + 1
                output = gcoc.graph_cut(new_data , self.img
                                        , np.array(self.labels.v) 
                                        , self.adj.v , self.data.length() 
                                        , mode , sigma , bias , new_data.max()+1)

                # second round takes care of holes with with
                # stochastic algorithm, since hole filling fails with
                # a non-stochastic process
                ubg, ufg, phi, num_comp, num_hole = \
                    topocut.topofix.fix(self.data.regions[0].astype('float64')
                                        , self.data.regions[1].astype('float64')
                                        , output,-1,0, stocastic=True)

                self.data.regions[0] = ubg.astype('int16')
                self.data.regions[1] = ufg.astype('int16')

                new_data = self.data.matrix()

                output = gcoc.graph_cut(new_data , self.img
                                        , np.array(self.labels.v) 
                                        , self.adj.v , self.data.length()
                                        , mode , sigma , bias , new_data.max()+1)

                num_hole = label.num_components(output,full=False)[0] - 1
                num_comp = label.num_components(output,full=False)[1]

        # ignore bg if mask is defined
        if( (max(label.num_components(output,full=False)) > 1)
            if self.mask is None else 
            (max(label.num_components(output,full=False)[1:]) > 1) ):
            print('ERROR: Inconsistent inter-segment topology')
        # fully reinitialize self (recompute adj, data, etc.)
        self.__init__(self.img
                      , output
                      , self.shifted
                      , self.win
                      , self.mask
                      , lightweight=lite
                      , nodata=lite
                      , center=self.center)
        return self.labels.v
        # return output

    def copy(self):
        cp = Slice(self.img.copy()
                   , self.labels.v.copy()
                   , self.shifted
                   , self.win
                   , self.mask
                   , lightweight=True
                   , nodata=True
                   , center=self.center
                   , bg=self.bg)
        cp.data = self.data.copy()
        return cp
