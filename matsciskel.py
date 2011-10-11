#!/usr/bin/python2.7
import sys,os,cv
import numpy as np
sys.path.append(os.getcwd());
import gco

inimg = cv.LoadImageM("seq3/img/image0041.png")
im = cv.CreateMat(inimg.rows, inimg.cols, cv.CV_8U)
cv.CvtColor(inimg,im, cv.CV_RGB2GRAY)

seed=np.genfromtxt("seq3/labels/image0040.label")

seed[seed!=3]=0;
seed[seed==3]=255;

# im = np.asarray( im[:,:] )      # convert to numpy array

print im.shape
print seed.shape

# print gco.graph_cut(a,a,a,a,2)
