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

def watershed_fixed(im,labels,d=0,suppression=3,intensity=False):
    import pymorph

    def grad_mag(img):
        x,y = np.gradient(img.astype('float'))
        return np.clip(
            np.round( 
                np.sqrt(
                    np.square(x)+np.square(y)) 
                )
            , 0
            , 255
            ).astype('uint8')

    if d > 0:
        new_labels = np.zeros_like(labels)
        for l in range(1,labels.max()+1):
            new_labels[erode_by(labels==l,
                                d,
                                min_size=1)] = l
        labels=new_labels

    # scipy.misc.imsave("dilate_test.png", labels_to_edges(labels))
    im = pymorph.hmin(im if not intensity else grad_mag(im),suppression)
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

def line(p,shape,r):
    return ndimage.morphology.distance_transform_edt(
        np.logical_not(
            bresenham((p[1],p[0],p[3],p[2]),shape))) < r

def bresenham(p,shape):
    """typical implementation of Bresenham's algorithm, augmented to
    work with numpy arrays"""
    """modified from: http://roguebasin.roguelikedevelopment.org/index.php/Bresenham%27s_Line_Algorithm """
    x1,y1,x2,y2 = p[:4]
    a = np.zeros(shape,dtype='bool')
    issteep = abs(y2-y1) > abs(x2-x1)
    if issteep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
    deltax = x2 - x1
    deltay = abs(y2-y1)
    error = int(deltax / 2)
    y = y1
    ystep = None
    if y1 < y2:
        ystep = 1
    else:
        ystep = -1
    for x in range(x1, x2):
        if issteep:
            if x >= 0 and x < a.shape[0] and y >= 0 and y < a.shape[1]:
                a[x,y] = True
        else:
            if x >= 0 and x < a.shape[1] and y >= 0 and y < a.shape[0]:
                a[y,x] = True
        error -= deltay
        if error < 0:
            y += ystep
            error += deltax
    return a

def dilate(img,d):
    # Ugly hack to convert to and from a uint8 to circumvent opencv limitations
    return cv2.morphologyEx(convert_to_uint8(img),
                            cv2.MORPH_DILATE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                      (int(d),int(d)))).astype(bool)

def erode(img,d):
    return cv2.morphologyEx(convert_to_uint8(img),
                            cv2.MORPH_ERODE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                      (int(d),int(d)))).astype(bool)

def erode_to(img,d=3,rel_size=0.5,min_size=15):
    """erode to structure to specified relative size"""
    prev_size = np.count_nonzero(img)
    if(prev_size <= 0):
        return img
    new_size = prev_size
    center = ndimage.measurements.center_of_mass(img)
    center = (int(center[0]),int(center[1]))
    while new_size > rel_size * prev_size and new_size > min_size:
        img = erode(img,d)
        new_size = np.count_nonzero(img)

    # handle case where region was > min_size but was dilated to 0
    # anyway by using center pixel as result
    if new_size < 1:
        print("ERROR: region dilated to size 0")
        img2 = np.zeros_like(img).astype('bool')
        img2[center] = True
        return img2
    else:
        return img

def erode_by(img, iterations=3, min_size=15, d=3):
    """erode to structure by numiterations or min size"""
    prev_size = np.count_nonzero(img)
    if(prev_size <= min_size):
        return img
    new_size = prev_size
    center = ndimage.measurements.center_of_mass(img)
    center = (int(center[0]),int(center[1]))
    for i in range(1,iterations):
        img = erode(img,d)
        new_size = np.count_nonzero(img)
        if new_size < min_size:
            break

    # handle case where region was > min_size but was dilated to 0
    # anyway by using center pixel as result
    if new_size < 1:
        img2 = np.zeros_like(img).astype('bool')
        img2[center] = True
        return img2
    else:
        return img

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

# def fit_log(im,a,r,g):
#     """get negative log liklihood from a gaussian mean"""
#     out = r.copy()
#     out[a] = np.log(np.absolute( 255-(im[a] - g[0]) ))
#     out[np.isinf(out)] = -1
#     return out

def fit_log(im,t,g):
    """get negative log liklihood from a gaussian mean"""
    diff = np.clip(np.absolute(im.astype('float') - float(g[0])) - t*g[1],0,255)
    return (-1.0*np.log( 1.0-diff/256.0 )).astype('float')
    # out = r.copy().astype('float')
    # # out[a] = np.log(np.absolute( 255-(im[a] - g[0]) ))
    # out[a] = -1.0*np.log( 1.0-np.absolute(im[a].astype('float') - float(g[0]))/256.0 )
    # # out[np.isinf(out)] = -1
    # return out

def bool_to_int16(b):
    output = np.zeros(b.shape,dtype='int16')
    output[b==0] = -1
    output[b>0] = 0
    return output

def bool_to_uint8(b):
    output = np.zeros(b.shape,dtype='uint8')
    output[b==0] = 0
    output[b>0] = 255
    return output

class Data(object):
    def __init__(self,labels=None):
        if not labels is None:
            self.regions = layer_list(labels)

    def convert_to_int16(self):
        self.regions = map(bool_to_int16 
                           , self.regions)
        
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

    def fit_log(self,im,d,t):
        from functools import partial
        convert_to_int = lambda d,a: ((a/d)*255).astype('int16')
        f = partial(fit_log,im,t)
        print("Computing log band area")
        area = map(lambda a : dilate(a,d), self.regions)
        print("Log fit")
        # no stddev for bg
        likelihoods = map(f,[(0.0,0.0)] + compute_gaussian(self.regions[1:],im))
        likelihoods = map(
            partial(
                convert_to_int
                , max(map(np.max,likelihoods)))
            , likelihoods)
        for i in range(len(likelihoods)) :
            likelihoods[i][np.logical_not(area[i])] = -1
        self.regions = likelihoods

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
        from label import largest_connected_component
        self.regions = [self.regions[0]] + \
            map(largest_connected_component,self.regions[1:])

        # close holes in the data term
        self.regions = [self.regions[0]] + \
            map(lambda (l): erode(dilate(l,tolerance),tolerance),
                self.regions[1:])

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
            # hack: add fixed amount to push it past the border
            return int(max(min(dist,max_d),min_d))+5 
        def find_max_dist(im1,im2):
            dist = ndimage.morphology.distance_transform_edt(np.logical_not(im1))
            return dist[im2].max()
        edges = labels_to_edges(watershed_fixed(im.copy(),labels.v.copy(),w_d,w_s))
        # for img in self.regions:
        #     print(clamp(find_max_dist(edges,pre(img))))
        self.regions = map(lambda img :
                        dilate(img,clamp(find_max_dist(edges,pre(img)))),
                        self.regions)

    def dilate_fixed_center(self,d,rel_size=0.25,min_size=15,first=False):
        if first:
            tmp = [erode_to(img,3,rel_size,min_size) for img in self.regions]
            self.regions = [ dilate(img,d) for img in self.regions ]
            for x in zip(range(len(self.regions)),tmp):
                self.label_exclusive(*x)
        else:
            tmp = [np.zeros_like(self.regions[0])] + \
                [erode_to(img,3,rel_size,min_size) for img in self.regions[1:]]
            self.regions = [self.regions[0]] + \
                [ dilate(img,d) for img in self.regions[1:] ]
            for x in zip(range(len(self.regions)),tmp):
                self.label_exclusive(*x)

    def output_data_term2(self):
        # just for testing (spit out data term as images)
        for i in range(0,len(self.regions)):
            output = self.regions[i]
            output[output==False] = 0
            output[output==True] = 255
            # scipy.misc.imsave("seq5/output/data"+str(i)+".png",output)
            scipy.misc.imsave("d"+str(i)+".png",output)

    def output_data_term(self,fname='data.png'):
        output = np.zeros(self.regions[0].shape,dtype='uint8')
        def alpha_composite(a,b,alpha=0.5):
            return np.add(np.multiply(a,alpha).astype('uint8'),
                          np.multiply(b,1-alpha).astype('uint8'))
        def combine(a,b):
            return np.add(a,np.divide(b,4))
        for i in range(0,len(self.regions)):
            s = self.regions[i].astype('uint8')
            s[s==False] = 0
            s[s==True] = 255
            output = combine(output,s)
            # scipy.misc.imsave("seq5/output/data"+str(i)+".png",output)
        scipy.misc.imsave(fname,output)
        return output

    def add_dummy_label(self):
        self.regions += [np.ones_like(self.regions[-1])]

    def label_erase(self,l):
        self.regions[l] = np.zeros_like(self.regions[l])

    def pixels_exclusive(self,pixels):
        for i,j,l in pixels:
            for r in range(0,len(self.regions)):
                self.regions[r][i,j] = True if r==l else False

    def label_exclusive(self,l,reg=None):
        if reg is None:
            reg = self.regions[l]
        self.regions = [ np.logical_and(np.logical_not(reg),x[1])
                      if x[0]!=l else x[1]
                      for x in zip(range(len(self.regions)),self.regions)]

    def label_inexclusive(self,reg,l):
        self.regions = [ np.logical_or(reg,x[1])
                      if x[0]!=l else x[1]
                      for x in zip(range(len(self.regions)),self.regions)]

    def or_all(self,reg,skip=[]):
        self.regions = [ np.logical_or(reg,x[1])
                         if not x[0] in skip else x[1]
                         for x in zip(range(len(self.regions)),self.regions)]

    def or_term(self,reg):
        self.regions = [self.regions[0]] + \
            map(lambda img : np.logical_or(img,reg), self.regions[1:])

    def matrix(self):
        return stack_matrix(self.regions)
    
    def length(self):
        return(len(self.regions))

    def copy(self):
        from copy import deepcopy
        cp = Data()
        cp.regions = deepcopy(self.regions)
        return cp
