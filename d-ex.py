#!/usr/bin/env python
import sys,os
import numpy as np
from figure import Score
import cPickle as pickle

def read_score(path,sl,d=3):
    scores = np.genfromtxt(path,dtype='float',delimiter=',')[d-1]
    return Score(sl
                 , scores[0]
                 , scores[1]
                 , scores[2]
                 , scores[3])

def main(*args):
    dimensions = (11,50)
    m = np.zeros(dimensions,dtype='float')

    for (j,i), v in np.ndenumerate(m):
        m[j,i] = read_score('seq1/d-ex/{0}/image{1:04d}.score'.format(i+1,j+90),j).f

	pickle.dump(m,open('d-ex.pkl','w'))

if __name__ == '__main__':
    sys.exit(main(*sys.argv))
