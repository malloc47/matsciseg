#data module
import numpy as np
import scipy

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
    self.regions = layer_list(self.labels)
