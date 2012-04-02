# gco module
import sys,os,cv,cv2
import numpy as np
sys.path.insert(0,os.getcwd())
import gcoc
import scipy
from scipy import ndimage
import gui

def unstack_matrix(layered):
    """splits 3D matrix to a list of 2D matrices"""
    unstacked = []
    for i in  range(layered.shape[2]) :
        unstacked.append(layered[:,:,i])
    return unstacked

def stack_matrix(l):
    """ 
    converts list of matrices to a single high-dimension matrix
    """
    stack = l[0]
    for i in  range(1,len(l)) :
        stack = np.dstack((stack,l[i]))
    return stack

def check_data_term(l):
    """make sure every pixel has at least one label"""
    return np.all(reduce(np.logical_or,l))
	

def convert_to_uint8(mat,maxval=255):
    newmat = np.zeros(mat.shape,dtype='uint8')
    newmat[np.nonzero(mat)] = 255;
    return newmat

def select_region(mat,reg):
    """returns binary matrix of a specific region in a label"""
    labels = np.zeros(mat.shape,dtype=bool)
    labels[mat==reg] = True
    return labels

def select_region_uint8(mat,reg,maxval=255):
    """selects region, but returns a uint8, which can be saved"""
    labels = mat.copy()
    labels[labels!=reg]=-1
    labels[labels==reg]=maxval
    labels[labels==-1]=0
    return labels.astype('uint8')

def layer_op(layered, op, dofirst=True):
    """Modifies, in place, the layered image with op; can ignore the
    first layer (0) if desired"""
    for i in range(layered.shape[2]) if dofirst else range(1,layered.shape[2]):
        layered[:,:,i] = op(layered[:,:,i])

def layer_matrix(unlayered):
    """convert a matrix of integer labels to a 3D binary matrix """
    data = select_region(unlayered,0)
    num_labels = int(unlayered.max()+1)
    for l in range(1,num_labels) :
        label = select_region(unlayered,l)
        data = np.dstack((data,label))
    return data

def layer_list(unlayered):
    """convert a matrix of integer labels to a list of 2D matrices"""
    data = [select_region(unlayered,0)]
    num_labels = int(unlayered.max()+1)
    for l in range(1,num_labels) :
        data.append(select_region(unlayered,l))
    return data

def skel(img):
    """skeletonize image"""
    print("Skel")
    hits = [ np.array([[0, 0, 0], [0, 1, 0], [1, 1, 1]]),
             np.array([[0, 0, 0], [1, 1, 0], [0, 1, 0]]) ]
    misses = [ np.array([[1, 1, 1], [0, 0, 0], [0, 0, 0]]),
               np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]]) ]

    for i in range(6):
        hits.append(np.transpose(hits[-2])[::-1, ...])
        misses.append(np.transpose(misses[-2])[::-1, ...])

    filters = zip(hits,misses)

    while True:
        prev = img
        for hit, miss in filters:
            filtered = ndimage.binary_hit_or_miss(img, hit, miss)
            img = np.logical_and(img, np.logical_not(filtered))
        if np.abs(prev-img).max() == 0:
            break
    return img

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

def dilate(img,d):
    # Ugly hack to convert to and from a uint8 to circumvent opencv limitations
    return cv2.morphologyEx(convert_to_uint8(img),
                            cv2.MORPH_DILATE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                      (d,d))).astype(bool)

def erode(img,d):
    return cv2.morphologyEx(convert_to_uint8(img),
                            cv2.MORPH_ERODE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                      (d,d))).astype(bool)

def relative_complement(p):
    return np.logical_and(p[0],np.logical_not(p[1]))

def region_clean(regions):
    out = np.zeros(regions.shape,dtype=regions.dtype)
    for l in range(regions.max()+1) :
        layer = (regions==l)
        if layer.any() :
            labeled = ndimage.label(layer>0)[0]
            # copy only the largest connected component
            out[np.nonzero(labeled == label_max(labeled))] = l
    # todo: what's happening with the zeros?
    return out

def compute_gaussian(layers,img):
    """obtain mean and std dev for all layers"""
    def layer_gaussian(l):
        vals = img[np.nonzero(l)]
        return (scipy.mean(vals),scipy.std(vals))
    return map(layer_gaussian,layers)

def fit_gaussian(v,g0,g1,d):
    """determine if v falls within d*g1(std dev) of mean g0"""
    if(np.isnan(v) or np.isnan(g0) or np.isnan(g1)): return False
    return ((v > g0-d*g1) and (v < g0+d*g1))

def smoothFn(s1,s2,l1,l2,adj):
    """smoothness function that could be passed to the minimzation"""
    if l1==l2 :
        return 0
    if not adj :
        return 10000000
    return int(1.0/(max(float(s1),float(s2))+1.0) * 255.0)
    # return int(1.0/float((abs(float(s1)-float(s2)) if \
        # abs(float(s1)-float(s2)) > 9 else 9)+1)*255.0)

class Volume(object):
    def __init__(self, img, labels, shifted={}, win=(0,0), mask=None):
        """initialize fields and compute defaults"""
        # These values are created
        # when the class is instantiated.
        self.img = img.copy()
        self.labels = region_clean(region_shift(labels,
                                                region_transform(labels)))
        self.orig_labels = self.labels.copy()
        # self.num_labels = self.labels.max()+1
        self.data = layer_list(self.labels)
        self.orig = np.array(self.data)
        self.adj = adjacent(self.labels)
        self.shifted=shifted
        self.win=win
        self.shifted=shifted
        self.mask=mask

    def skel(self):
        """run skeletonization and integrate to data term"""
        # sk = [self.data[0]] + \
        #     map(lambda img : erode(skel(img),10), self.data[1:])
        sk = map(lambda img : erode(skel(img),1), self.orig)
        for i in range(0,len(sk)):
            s = sk[i];
            for k in range(0,len(self.data)):
                if k == i:
                    # add skeltonization to its own data term
                    self.data[k] = np.logical_or(self.data[k],s)
                else:
                    # remove skeletonization from all other terms
                    self.data[k] = relative_complement((self.data[k],s))

    def fit_gaussian(self,d,d2):
        """
        compute and fit gaussian on all pixels within a band radius
        d/2 around a label, where the fit is within d2 std deviations
        """
        # (erosion,dilation)
        area = map(relative_complement,
                   zip(map(lambda img : dilate(img,d), self.data[1:]),
                       map(lambda img : erode(img,d), self.data[1:])))

        fit = np.vectorize(fit_gaussian)

        print(compute_gaussian(self.data[1:],self.img))

        combine = map(lambda (a,g):
                      erode(dilate(np.logical_and(a,fit(self.img,g[0],g[1],d2)),5),1),
                      zip(area,compute_gaussian(self.data[1:],self.img)))

        self.data = [self.data[0]] + \
            map(lambda (d,n): np.logical_or(d,n),zip(self.data[1:],combine))
        
        # redo matrix
        self.data[0] = dilate(np.logical_not(reduce(np.logical_or,self.data[1:])),20) #,5)

    def dilate(self,d):
        """dilate all but first (background) region"""
        self.data = [self.data[0]] + \
            map(lambda img : dilate(img,d), self.data[1:])

    def dilate_all(self,d):
        """dilate all regions"""
        self.data = map(lambda img : dilate(img,d), self.data)

    def dilate_label(self,l,d):
        """dilate specific regions"""
        self.data[l] = dilate(self.data[l],d)

    def dilate_first(self,d):
        """dilate first (background) region only"""
        self.data[0] = dilate(self.data[0],d)

    def output_data_term(self):
        # just for testing (spit out data term as images)
        for i in range(0,len(self.data)):
            output = self.data[i]
            output[output==False] = 0
            output[output==True] = 255
            # scipy.misc.imsave("seq5/output/data"+str(i)+".png",output)
            scipy.misc.imsave("d"+str(i)+".png",output)

    def get_adj(self,label_list):
        """return all labels that are adjacent to label_list labels"""
        output = []+label_list  # include original labels as well
        for l in label_list:
            output += [i for i,x in enumerate(self.adj[l,:]) if x]
        return set(output)

    def get_adj_radius(self,p):
        output = set()  # include original labels as well
        for i,v in np.ndenumerate(self.labels):
            if gui.dist(i,p[0:2]) < p[2]:
                output.add(self.labels[i])
        return output

    def label_exclusive(self,l):
        self.data = [ np.logical_and(np.logical_not(self.labels==l),x[1])
                      if x[0]!=l else x[1]
                      for x in zip(range(len(self.data)),self.data)]

    def label_inexclusive(self,l):
        self.data = [ np.logical_or(self.labels==l,x[1])
                      if x[0]!=l else x[1]
                      for x in zip(range(len(self.data)),self.data)]

    def set_adj_all(self):
        """remove all topology constraints (everything adjacent)"""
        self.adj[:,:] = True

    def set_adj_label_all(self,l):
        """remove topology constraints for a single label"""
        self.adj[l,:] = True
        self.adj[:,l] = True

    def edit_labels(self,d):
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

    def remove_label(self,l,d):
        v = self.crop([l])
        v.dilate_all(d)
        v.label_inexclusive(l)
        v.set_adj_all()
        v.graph_cut_no_clean()
        return v

    def add_label(self,p,d):
        v = self.crop(list(self.get_adj_radius(p)))
        p = (p[0]-v.win[0],p[1]-v.win[1],p[2])
        l = v.add_label_circle(p)
        v.dilate_label(l,p[2]*d)
        v.label_exclusive(l)
        v.output_data_term()
        v.set_adj_label_all(l)
        v.graph_cut_no_clean()
        return v

    def add_label_circle(self,p):
        new_label = self.labels.max()+1
        for i,v in np.ndenumerate(self.labels):
            if gui.dist(i,p[0:2]) < p[2]:
                self.labels[i] = new_label
        # reconstruct data term after adding label
        self.data = layer_list(self.labels)
        self.adj = adjacent(self.labels)
        self.__init__(self.img,
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
        return Volume(new_img,new_seed,label_transform,(y0,x0),mask_cropped)

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
                new_label+=1

        # self.__init__(self.img,
        #               region_clean(region_shift(self.labels,
        #                                         region_transform(self.labels))),
        #               self.shifted,
        #               self.win)

    def graph_cut(self,mode=0):
        """run graph cut on this volume (mode specifies V(p,q) term"""
        # just for testing (spit out data term as images)
        # if self.win != (0,0):
        # for i in range(0,len(self.data)):
        #     output = self.data[i]
        #     output[output==False] = 0
        #     output[output==True] = 255
        #     # scipy.misc.imsave("seq5/output/data"+str(i)+".png",output)
        #     scipy.misc.imsave("d"+str(i)+".png",output)

        # w=gui.Window(self.img,self.labels)

        if not check_data_term(self.data):
            print("Not all pixels have a label")
        else:
            print("All pixels have a label")

        output = gcoc.graph_cut(stack_matrix(self.data),
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
        output = gcoc.graph_cut(stack_matrix(self.data),
                                self.img,
                                self.labels,
                                self.adj,
                                self.labels.max()+1, # todo: extract from data
                                mode)
        self.labels = output.copy()
        return self.labels
