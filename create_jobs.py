#!/usr/bin/env python
import sys,os
sys.path.insert(0,os.path.join(os.getcwd(),'..'))
import numpy as np
import scipy
import cPickle as pickle
import matsciskel
import matsci

def main(*args):
    if(len(args) < 1):
        return 1

    datasets = []
    names = [ 'd1s'+str(i) for i in range(1,13) ]
    name_to_dir = lambda n: 'syn/'+n
    stdlen = range(0,50)
    ground_slices = [5,15,25,35,45]

    dilation = 10
    def global_cmd(n,rn,i,j,r):
        if rn==i:
            return """ mkdir -p {0}/{4}/{1:d}/
ln -s ../../ground/{2:04d}.label {0}/{4}/{1:d}/{2:04d}.label
./matsciskel.py e global {0}/img/{2:04d}.png {0}/ground/{2:04d}.label {0}/img/{3:04d}.png {0}/{4}/{1:d}/{3:04d}.label {0}/{4}/{1:d}/{3:04d}.png {5}
""".format(n,rn,i,j,r,dilation)
        else:
            return """./matsciskel.py e global {0}/img/{2:04d}.png {0}/{4}/{1:d}/{2:04d}.label {0}/img/{3:04d}.png {0}/{4}/{1:d}/{3:04d}.label {0}/{4}/{1:d}/{3:04d}.png {5}
""".format(n,rn,i,j,r,dilation)
    datasets += [ (n,'global-'+str(dilation),stdlen,ground_slices,global_cmd) for n in names]

    def global_local_cmd(n,rn,i,j,r):
        if rn==i:
            return """mkdir -p {0}/{4}/{1:d}/
ln -s ../../ground/{2:04d}.label {0}/{4}/{1:d}/{2:04d}.label
./matsciskel.py e global {0}/img/{2:04d}.png {0}/ground/{2:04d}.label {0}/img/{3:04d}.png {0}/{4}/{1:d}/{3:04d}-old.label {0}/{4}/{1:d}/{3:04d}.png {5}
LD_LIBRARY_PATH=/home/malloc47/src/programs/OpenCV-2.0.0/build/lib /home/malloc47/src/projects/matsci/matscicut-debian/matscicut {0}/img/{3:04d}.png {0}/{4}/{1:d}/{3:04d}-old.label {0}/{4}/{1:d}/{3:04d}.label
""".format(n,rn,i,j,r,dilation)
        else:
            return """./matsciskel.py e global {0}/img/{2:04d}.png {0}/{4}/{1:d}/{2:04d}.label {0}/img/{3:04d}.png {0}/{4}/{1:d}/{3:04d}-old.label {0}/{4}/{1:d}/{3:04d}.png {5}
LD_LIBRARY_PATH=/home/malloc47/src/programs/OpenCV-2.0.0/build/lib /home/malloc47/src/projects/matsci/matscicut-debian/matscicut {0}/img/{3:04d}.png {0}/{4}/{1:d}/{3:04d}-old.label {0}/{4}/{1:d}/{3:04d}.label
""".format(n,rn,i,j,r,dilation)
    datasets += [ (n,'global-local-'+str(dilation),stdlen,ground_slices,global_local_cmd) for n in names]

    def local_cmd(n,rn,i,j,r):
        if rn==i:
            return """mkdir -p {0}/{4}/{1:d}/
ln -s ../../ground/{2:04d}.label {0}/{4}/{1:d}/{2:04d}.label
./matsciskel.py e clique {0}/img/{2:04d}.png {0}/ground/{2:04d}.label {0}/img/{3:04d}.png {0}/{4}/{1:d}/{3:04d}.label {0}/{4}/{1:d}/{3:04d}.png {5}
""".format(n,rn,i,j,r,dilation)
        else:
            return """./matsciskel.py e clique {0}/img/{2:04d}.png {0}/{4}/{1:d}/{2:04d}.label {0}/img/{3:04d}.png {0}/{4}/{1:d}/{3:04d}.label {0}/{4}/{1:d}/{3:04d}.png {5}
""".format(n,rn,i,j,r,dilation)
    datasets += [ (n,'local-'+str(dilation),stdlen,ground_slices,local_cmd) for n in names]

    def local_local_cmd(n,rn,i,j,r):
        if rn==i:
            return """mkdir -p {0}/{4}/{1:d}/
ln -s ../../ground/{2:04d}.label {0}/{4}/{1:d}/{2:04d}.label
./matsciskel.py e clique {0}/img/{2:04d}.png {0}/ground/{2:04d}.label {0}/img/{3:04d}.png {0}/{4}/{1:d}/{3:04d}-old.label {0}/{4}/{1:d}/{3:04d}.png {5}
LD_LIBRARY_PATH=/home/malloc47/src/programs/OpenCV-2.0.0/build/lib /home/malloc47/src/projects/matsci/matscicut-debian/matscicut {0}/img/{3:04d}.png {0}/{4}/{1:d}/{3:04d}-old.label {0}/{4}/{1:d}/{3:04d}.label
""".format(n,rn,i,j,r,dilation)
        else:
            return """./matsciskel.py e clique {0}/img/{2:04d}.png {0}/{4}/{1:d}/{2:04d}.label {0}/img/{3:04d}.png {0}/{4}/{1:d}/{3:04d}-old.label {0}/{4}/{1:d}/{3:04d}.png {5}
LD_LIBRARY_PATH=/home/malloc47/src/programs/OpenCV-2.0.0/build/lib /home/malloc47/src/projects/matsci/matscicut-debian/matscicut {0}/img/{3:04d}.png {0}/{4}/{1:d}/{3:04d}-old.label {0}/{4}/{1:d}/{3:04d}.label
""".format(n,rn,i,j,r,dilation)
    datasets += [ (n,'local-local-'+str(dilation),stdlen,ground_slices,local_local_cmd) for n in names]

    for n,run,slices,gt,cmd in datasets:
        for g in gt:
            forward =  [ e for e in slices if e>=g]
            backward = list(reversed([ e for e in slices if e<=g]))
            forward_file = 'ctrl/'+n+'-'+run+'-'+str(g)+'-f.sh'
            backward_file = 'ctrl/'+n+'-'+run+'-'+str(g)+'-b.sh'
            with open(forward_file, 'wb') as f:
                f.write('#!/bin/bash\n')
                for i, j in zip(forward[:-1], forward[1:]):
                    f.write(cmd(name_to_dir(n),g,i,j,run))
            with open(backward_file, 'wb') as f:
                f.write('#!/bin/bash\n')
                for i, j in zip(backward[:-1], backward[1:]):
                    f.write(cmd(name_to_dir(n),g,i,j,run))

# ./matsciskel.py e global seq1/img/image0090.png seq1/ground/image0090.label seq1/img/image0091.png test2.label test2.png 10

if __name__ == '__main__':
    sys.exit(main(*sys.argv))
