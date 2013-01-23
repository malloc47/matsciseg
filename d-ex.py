#!/usr/bin/env python
import sys,os
import numpy as np
import cPickle as pickle

import matplotlib
matplotlib.use('pdf')
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt

from figure import Score

def read_score(path,sl,d=3):
    scores = np.genfromtxt(path,dtype='float',delimiter=',')[d-1]
    return Score(sl
                 , scores[0]
                 , scores[1]
                 , scores[2]
                 , scores[3])

def main(*args):
    dimensions = (11+1,50+1)

    try:
        m = pickle.load(open('d-ex.pkl','r'))
    except IOError as e:
        m = np.zeros(dimensions,dtype='float')

        for (x,y), v in np.ndenumerate(m):
            if x > 0 and y > 0:
                m[x,y] = read_score('seq1/d-ex/{0}/image{1:04d}.score'.format(x+1,y+90),y).f

        pickle.dump(m,open('d-ex.pkl','w'))

    m = m[0:dimensions[0], 0:dimensions[1]]

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.view_init(20,-140)

    X, Y = np.meshgrid(range(1,dimensions[1]), range(1,dimensions[0]))
    # zs = np.array([m[x,y] for x,y in zip(np.ravel(X), np.ravel(Y))])
    surf = ax.plot_surface(X,Y,m,rstride=1,cstride=1,cmap=cm.jet,linewidth=0,antialiased=False)
    ax.set_xlabel('d')
    ax.set_ylabel('Slice')
    ax.set_zlabel('F-measure')
    ax.set_xlim(1, 50)

    # plt.show()
    plt.savefig('d-ex.pdf',bbox_inches='tight')
    print("done plotting")

if __name__ == '__main__':
    sys.exit(main(*sys.argv))
