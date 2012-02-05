# gco module
import sys,os,cv,cv2
import numpy as np
sys.path.insert(0,os.getcwd())
import gcoc
import scipy
from scipy import ndimage

def unstack_matrix(layered):
    unstacked = []
    for i in  range(layered.shape[2]) :
        unstacked.append(layered[:,:,i])
    return unstacked

def stack_matrix(l):
    stack = l[0]
    for i in  range(1,len(l)) :
        stack = np.dstack((stack,l[i]))
    return stack

def convert_to_uint8(mat,maxval=255):
    newmat = np.zeros(mat.shape,dtype='uint8')
    newmat[np.nonzero(mat)] = 255;
    return newmat

def select_region(mat,reg):
    labels = np.zeros(mat.shape,dtype=bool)
    labels[mat==reg] = True
    return labels

def select_region_uint8(mat,reg,maxval=255):
    labels = mat.copy()
    labels[labels!=reg]=-1
    labels[labels==reg]=maxval
    labels[labels==-1]=0
    return labels.astype('uint8')

# Modifies, in place, the layered image with op
# Can ignore the first layer (0) if desired
def layer_op(layered, op, dofirst=True):
    for i in  range(layered.shape[2]) if dofirst else range(1,layered.shape[2]):
        layered[:,:,i] = op(layered[:,:,i])

def layer_matrix(unlayered):
    data = select_region(unlayered,0)
    num_labels = int(unlayered.max()+1)
    for l in range(1,num_labels) :
        label = select_region(unlayered,l)
        data = np.dstack((data,label))
    return data

def layer_list(unlayered):
    data = [select_region(unlayered,0)]
    num_labels = int(unlayered.max()+1)
    for l in range(1,num_labels) :
        data.append(select_region(unlayered,l))
    return data

def skel(img):
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
    return reduce(np.logical_or,map(lambda l:labels==l,label_list))

def label_max(labels):
    sizes = []
    # check only indices above 0
    for l in range(1,labels.max()+1) :
        sizes.append((labels == l).sum())
    # increment index, as we ignored zero before
    return sizes.index(max(sizes))+1

def label_sizes(labels):
    sizes = []
    for l in range(0,labels.max()+1) :
        sizes.append((labels == l).sum())
    return sizes

def label_empty(labels):
    return [i for i,x in enumerate(label_sizes(labels)) if x == 0]

def region_transform(labels):
    unshifted = range(0,labels.max()+1)
    shifted = range(0,labels.max()+1)
    counter = 0
    empty = label_empty(labels)
    for l in unshifted :
        if l in empty :
            shifted[l] = -1
        else :
            shifted[l] = counter
            counter = counter + 1
    return zip(unshifted,shifted)

def region_shift(regions,transform) :
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
    # out = np.ones(regions.shape,dtype=regions.dtype) * -1
    out = np.zeros(regions.shape,dtype=regions.dtype)
    for l in range(regions.max()+1) :
        layer = (regions==l)
        if layer.any() :
            labeled = ndimage.label(layer>0)[0]
            # copy only the largest connected component
            out[np.nonzero(labeled == label_max(labeled))] = l
    return out

def compute_gaussian(layers,img):
    def layer_gaussian(l):
        vals = img[np.nonzero(l)]
        return (scipy.mean(vals),scipy.std(vals))
    return map(layer_gaussian,layers)

def fit_gaussian(v,g0,g1,d):
    if(np.isnan(v) or np.isnan(g0) or np.isnan(g1)): return False
    return ((v > g0-d*g1) and (v < g0+d*g1))

def smoothFn(s1,s2,l1,l2,adj):
    if l1==l2 :
        return 0
    if not adj :
        return 10000000
    return int(1.0/(max(float(s1),float(s2))+1.0) * 255.0)
    # return int(1.0/float((abs(float(s1)-float(s2)) if \
        # abs(float(s1)-float(s2)) > 9 else 9)+1)*255.0)

class Volume(object):
    def __init__(self, img, labels):
        # These values are created
        # when the class is instantiated.
        self.img = img.copy()
        self.labels = region_clean(region_shift(labels,
                                                region_transform(labels)))
        self.num_labels = self.labels.max()+1
        self.data = layer_list(self.labels)
        self.orig = np.array(self.data)
        self.adj = adjacent(self.labels)

    def skel(self):
        # sk = [self.data[0]] + \
        #     map(lambda img : erode(skel(img),10), self.data[1:])
        sk = map(lambda img : erode(skel(img),1), self.orig)
        for i in range(0,len(sk)):
            s = sk[i];
            for k in range(0,len(self.data)):
                if k == i:
                    self.data[k] = np.logical_or(self.data[k],s)
                else:
                    self.data[k] = relative_complement((self.data[k],s))

    def fit_gaussian(self,d,d2):
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
        self.data = [self.data[0]] + \
            map(lambda img : dilate(img,d), self.data[1:])

    def dilate_all(self,d):
        self.data = map(lambda img : dilate(img,d), self.data)

    def dilate_first(self,d):
        self.data[0] = dilate(self.data[0],d)

    def get_adj(self,label_list):
        output = []+label_list
        for l in label_list:
            output += [i for i,x in enumerate(self.adj[l,:]) if x]
        return set(output)

    def crop(self,label_list):
        label_list = self.get_adj(label_list)
        mask = create_mask(self.seed,label_list)
        mask_win = argwhere(mask)
        (y0, x0), (y1, x1) = mask_win.min(0), mask_win.max(0) + 1 
        # crop out everything with the given window
        mask_cropped = mask[y0:y1, x0:x1]
        cropped_seed = self.seed[y0:y1, x0:x1]
        new_img = self.img[y0:y1, x0:x1]
        # transform seed img
        new_seed = zeros(cropped_seed.shape).astype('int16')
        label_transform = []
        new_label = 1
        new_seed[np.logical_not(mask_cropped)] = 0
        for l in label_list:
            label_transform.append((l,new_label))
            new_seed[cropped_seed==l] = new_label
            new_label += 1
        # todo: output the label_transform somewhere and do the reverse operation
        return Volume(new_img,new_seed)

    def graph_cut(self,mode=0):
        # just for testing (spit out data term as images)
        # for i in range(0,len(self.data)):
        #     output = self.data[i]
        #     output[output==False] = 0
        #     output[output==True] = 255
        #     # scipy.misc.imsave("seq5/output/data"+str(i)+".png",output)
        #     scipy.misc.imsave("data"+str(i)+".png",output)

        output = gcoc.graph_cut(stack_matrix(self.data),
                                self.img,
                                self.labels,
                                self.adj,
                                self.num_labels,
                                mode)
        return region_clean( region_shift(output,region_transform(output)) )
