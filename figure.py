#!/usr/bin/env python
import sys,os
from collections import namedtuple
import itertools
import numpy as np
sys.path.insert(0,os.path.join(os.getcwd(),'.'))
# import pylab, matplotlib
import matplotlib as mpl
mpl.use('pdf')
import matplotlib.pyplot as plt
import re

Score = namedtuple('Score', ['slice','f', 'p', 'r', 'd'])
Dataset = namedtuple('Dataset', ['name', 'method', 'scores'])

def read_score(path,d=3):
    scores = np.genfromtxt(path,dtype='float',delimiter=',')[d-1]
    # scores = np.array([np.random.normal(loc=0.0,scale=1),0.8,0.95,5])
    return Score(int(os.path.splitext(re.sub('[^0-9]','',os.path.basename(path)))[0])
                     , scores[0]
                     , scores[1]
                     , scores[2]
                     , scores[3])

def plot_job(job,field='f',fmt='-ro'):
    xs = [j.slice for j in job]
    ys = [j._asdict()[field] for j in job]
    return plt.plot(xs,ys,fmt)

def plot_jobs(jobs,field='f',fmt='-ro'):
    for j in jobs:
        plot_job(jobs[j],field,fmt)

def plot_dataset(dataset,run=None):
    color = ['r','b','g','y','c','m','w','k', 'r','b','g','y','c','m','w','k']
    lgd = []
    lgd_names = []
    for d,c in zip(dataset,color):
        if run is None:
            plot_jobs(d.scores,fmt='-'+c)
        else:
            plot_job(d.scores[run],fmt='-'+c)
        lgd += [plt.Rectangle((0, 0), 1, 1, fc=c)]
        lgd_names += [d.method]
    plt.legend(lgd,lgd_names)

def lsd(*l):
    return filter(lambda d: os.path.isdir(os.path.join(*((os.getcwd(),)+l+(d,)))),
                  os.listdir(os.path.join(*((os.getcwd(),)+l))))

def main(*args):
    # get ready for a fun series of list/dict comprehensions...
    datasets = filter(lambda ds: any(ds.scores.values()),
                      [ Dataset(name
                                , method
                                , { num: 
                                    sorted([ 
                            read_score(os.path.join(os.getcwd()
                                                    , 'syn'
                                                    , name
                                                    , method
                                                    , num
                                                    , s)
                                       , 3) 
                            for s in 
                            filter(lambda d : d.endswith('.score')
                                   ,os.listdir(os.path.join(os.getcwd()
                                                            , 'syn'
                                                            , name,method,num))) ]
                                           , key=lambda sc: sc.slice)
                                    for num in 
                                    os.listdir(os.path.join(os.getcwd()
                                                            , 'syn'
                                                            , name,method))}
                                )
                 for (name,method) in
                 sum([zip([n]*len(m),m) for (n,m) in 
                      # [(nm,os.listdir(os.path.join(os.getcwd(),'syn',nm))) 
                      [(nm,lsd('syn',nm)) 
                       # for nm in os.listdir(os.path.join(os.getcwd(),'syn'))] ], [])
                       for nm in lsd('syn')] ], [])
                 if method != 'ground' and method != 'img']
                      )

    plt.figure()
    plot_dataset([ d for d in datasets if d.name == 'd1s16' ],run='150')
    plt.xlim(0,150)
    plt.savefig('d1s16.pdf')

    plt.figure()
    plot_dataset([ d for d in datasets if d.name == 'd1s17' ],run='150')
    plt.savefig('d1s17.pdf')

    plt.figure()
    plot_dataset([ d for d in datasets if d.name == 'd1s18' ],run='150')
    plt.savefig('d1s18.pdf')

    # plot_jobs(datasets[0].scores)
    # p1 = plt.Rectangle((0, 0), 1, 1, fc="r")
    # p2 = plt.Rectangle((0, 0), 1, 1, fc="b")
    # plt.legend([p1,p2], ['cool data','more cool'])

if __name__ == '__main__':
    sys.exit(main(*sys.argv))
