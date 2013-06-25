import scipy
import numpy as np

def read_labels(path):
    return np.genfromtxt(path,dtype='int16')

def read_grey_as_rgb(imgin):
    im = scipy.misc.imread(imgin,flatten=True).astype('float32')
    im = np.divide(im,im.max())
    im = np.multiply(im,255)
    im = np.dstack((im,im,im))
    return im

def read_grey_as_rgb_unflattened(imgin):
    # im = scipy.misc.imread(imgin,flatten=True).astype('float32')
    im = scipy.misc.imread(imgin).astype('float32')
    im = np.divide(im,im.max())
    im = np.multiply(im,255).astype('uint8')
    # im = np.dstack((im,im,im))
    return im

def read_img(img_name):
    import cv
    im = cv.LoadImageM(img_name)
    im_gray = cv.CreateMat(im.rows, im.cols, cv.CV_8U)
    cv.CvtColor(im,im_gray, cv.CV_RGB2GRAY)
    # convert to numpy arrays
    im_gray = np.asarray(im_gray[:,:])
    im = np.asarray(im[:,:])
    return (im,im_gray)
