#!/usr/bin/python2.7
import sys,os,scipy
import numpy as np
import h5py

def main(*args):
    fname = str(args[1])
    prefix = str(args[2])
    postfix = str(args[3])
    f = h5py.File(fname,'r')
    l = len(f['Data'])
    m = len(f['Data/0'])
    print "slices: " + str(l)
    print "modes: " + str(m)
    print "type: " + str(f['Data/0/0'][:].dtype)
    for j in range(m):
        sys.stdout.write("mode " + str(m) + ": ")
        for i in range(l):
            sys.stdout.write(str(i) + " ")
            sys.stdout.flush()
            im = f['Data/' + str(i) + '/' + str(j)][:]
            scipy.misc.imsave(prefix + str(i).zfill(4) + "_" + str(j).zfill(2) + postfix,im)
        print ""
    f.close()

if __name__ == '__main__':
    sys.exit(main(*sys.argv))
