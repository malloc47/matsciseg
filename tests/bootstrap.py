import os,sys
sys.path.insert(0,os.path.join(os.getcwd()))
import matsci
import matsciskel
import numpy as np
import scipy
seed=np.genfromtxt('seq1/ground/image0090.label',dtype='int16')
im,img = matsciskel.read_img('seq1/img/image0090.png')
v = matsci.gco.Slice(img,seed)
