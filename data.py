#data module
import cv,cv2
import numpy as np
import scipy
from scipy import ndimage

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

def list_layer(layered):
    """convert a list of 2D matrices to a matrix of integer labels"""
    output = np.zeros(layered[0].shape).astype('int16')
    for l in range(0,len(layered)):
        output[layered[l]] = l
    return output


def labels_to_edges(labels):
    grad = np.gradient(labels)
    edges = np.maximum(abs(grad[0]),abs(grad[1]))
    return edges>0

def watershed(im,labels,d=0,suppression=3):
    import pymorph
    if d > 0:
        # hack: dilate edges instead of eroding structures
        edges = dilate(labels_to_edges(labels),d)
        # grad = np.gradient(labels)
        # edges = dilate(np.maximum(abs(grad[0]),abs(grad[1]))>0,d)
        labels[edges] = 0
    im = pymorph.hmin(im,suppression)
    return ndimage.watershed_ift(im, labels)
    
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

def fast_fit_gaussian(v,g,d):
    """faster function to determine if v falls within d*g1(std dev) of
    mean g0"""
    return np.logical_and(v>g[0]-g[1],v<g[0]+g[1])

class Data(object):
    def __init__(self,labels=None):
        if not labels is None:
            self.regions = layer_list(labels)

    def skel(self,d=None,erode_size=1):
        """run skeletonization and integrate to data term"""
        # allow another data term to be passed in
        if not d:
            d = self
        # sk = [self.regions[0]] + \
        #     map(lambda img : erode(skel(img),10), self.regions[1:])
        sk = map(lambda img : erode(skel(img),erode_size), d.regions)
        for i in range(0,len(sk)):
            s = sk[i];
            for k in range(0,len(self.regions)):
                if k == i:
                    # add skeltonization to its own data term
                    self.regions[k] = np.logical_or(self.regions[k],s)
                else:
                    # remove skeletonization from all other terms
                    self.regions[k] = relative_complement((self.regions[k],s))

    def fit_gaussian(self,im,d,d2,d3):
        """
        compute and fit gaussian on all pixels within a band radius
        d/2 around a label, where the fit is within d2 std deviations
        """
        tolerance = 2
        # (erosion,dilation)
        print("Computing gaussian band area")
        area = map(relative_complement,
                   zip(map(lambda img : dilate(img,d), self.regions[1:]),
                       map(lambda img : dilate(img,d2), self.regions[1:])))

        # fit = np.vectorize(fit_gaussian)

        # diagnostic
        # print(compute_gaussian(self.regions[1:],self.img))

        print("Gaussian fit")
        combine = map(lambda (a,g):
                      np.logical_and(a, fast_fit_gaussian(im,g,d3)),
                                     # fit(self.img,g[0],g[1],d3)),
                      zip(area,compute_gaussian(self.regions[1:],im)))

        print("Combining regions")
        self.regions = [dilate(self.regions[0],d2)] + \
            map(lambda (s,n): np.logical_or(dilate(s,d2),n),
                zip(self.regions[1:],combine))

        # fetch the largest component (remove unconnected straggling pixels)
        from gco import largest_connected_component
        self.regions = [self.regions[0]] + \
            map(largest_connected_component,self.regions[1:])

        # close holes in the data term
        self.regions = [self.regions[0]] + \
            map(lambda (l): erode(dilate(l,tolerance),tolerance),self.regions[1:])

        # redo matrix
        # self.regions[0] = dilate(np.logical_not(reduce(np.logical_or,self.regions[1:])),d) #,5)

    def dilate(self,d):
        """dilate all but first (background) region"""
        self.regions = [self.regions[0]] + \
            map(lambda img : dilate(img,d), self.regions[1:])

    def dilate_all(self,d):
        """dilate all regions"""
        self.regions = map(lambda img : dilate(img,d), self.regions)

    def dilate_label(self,l,d):
        """dilate specific regions"""
        self.regions[l] = dilate(self.regions[l],d)

    def dilate_first(self,d):
        """dilate first (background) region only"""
        self.regions[0] = dilate(self.regions[0],d)

    def dilate_auto(self,im,labels,max_d,min_d=1,w_d=10,w_s=3):
        def pre(im):
            return ndimage.morphology.morphological_gradient(im,size=(3,3))
        def clamp(dist):
            return int(max(min(dist,max_d),min_d))
        def find_max_dist(im1,im2):
            dist = ndimage.morphology.distance_transform_edt(np.logical_not(im1))
            return dist[im2].max()
        edges = labels_to_edges(watershed(im.copy(),labels.copy(),w_d,w_s))
        # for img in self.regions:
        #     print(clamp(find_max_dist(edges,pre(img))))
        self.regions = map(lambda img : 
                        dilate(img,clamp(find_max_dist(edges,pre(img)))), 
                        self.regions)

    def output_data_term2(self):
        # just for testing (spit out data term as images)
        for i in range(0,len(self.regions)):
            output = self.regions[i]
            output[output==False] = 0
            output[output==True] = 255
            # scipy.misc.imsave("seq5/output/data"+str(i)+".png",output)
            scipy.misc.imsave("d"+str(i)+".png",output)

    def output_data_term(self):
        output = np.zeros(self.regions[0].shape,dtype='uint8')
        def alpha_composite(a,b,alpha=0.5):
            return np.add(np.multiply(a,alpha).astype('uint8'),np.multiply(b,1-alpha).astype('uint8'))
        def combine(a,b):
            return np.add(a,np.divide(b,4))
        for i in range(0,len(self.regions)):
            s = self.regions[i].astype('uint8')
            s[s==False] = 0
            s[s==True] = 255
            output = combine(output,s)
            # scipy.misc.imsave("seq5/output/data"+str(i)+".png",output)
        scipy.misc.imsave("data.png",output)

    def label_exclusive(self,labels,l):
        self.regions = [ np.logical_and(np.logical_not(labels==l),x[1])
                      if x[0]!=l else x[1]
                      for x in zip(range(len(self.regions)),self.regions)]

    def label_inexclusive(self,labels,l):
        self.regions = [ np.logical_or(labels==l,x[1])
                      if x[0]!=l else x[1]
                      for x in zip(range(len(self.regions)),self.regions)]

    def matrix(self):
        return stack_matrix(self.regions)

    def copy(self):
        from copy import deepcopy
        cp = Data()
        cp.regions = deepcopy(self.regions)
        return cp
