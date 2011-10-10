#!/usr/bin/python2.7
import sys
sys.path.append("/home/malloc47/src/projects/matsci/matsciskel/gco");
# sys.path.append(".");
import cv
import numpy as np
import gco
im = cv.LoadImageM("seq3/img/image0041.png")
mat = cv.CreateMat( 3 , 3 , cv.CV_8U )
cv.Set( mat , 1 )
a = np.asarray( mat[:,:] )
# a[0,1] = 2;
# a[0,2] = 3;
# a[1,0] = 4;
# a[1,1] = 5;
# a[1,2] = 6;
# a[2,0] = 7;
# a[2,1] = 8;
# a[2,2] = 9;
print gco.graph_cut(a,a,a,a,2)
