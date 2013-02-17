#!/usr/bin/env python
import sys,os,cv,cv2
import numpy as np
import scipy
from scipy import ndimage
import recipes
import argparse
import inspect

edge_types = {
    'i' : 0,
    'e' : 1,
    'm' : 2,
    's' : 3,
    't' : 4,
    }

def run_jobs(pargs):
    for f in range(0,len(pargs.files),4):        
        im,im_gray = read_img(pargs.files[f])
        im2,im2_gray = read_img(pargs.files[f+2])
        labels1 = np.genfromtxt(pargs.files[f+1],dtype='int16')
        labels2 = np.genfromtxt(pargs.files[f+3],dtype='int16')
        # todo: wrap these in dict, use inspect to determine which
        # args to pass, and then pass all the arguments from pargs
        # plus these args to the function

def main(*args):
    parser = argparse.ArgumentParser(
        description='CLI for materials image segmentation propagation')
    # parser.add_argument('-b','--binary',
    #                     action='store',dest='binary_type',
    #                     choices=edge_types.keys(), default='e', 
    #                     help='Type of edges (binary term)')
    subparsers = parser.add_subparsers(dest='subcmd')

    cmds = recipes.cmds

    for cmdname, cmdval in cmds.iteritems():
        cmd_parser = subparsers.add_parser(cmdname)
        for argname, argval in cmdval['args'].iteritems():
            # the only "blessed" parameter is the name field
            namelst = argval['name'] if 'name' in argval else ['--'+argname]
            cmd_parser.add_argument(
                *namelst,
                 **{k:v for k,v in argval.iteritems() if k!='name'})

        cmd_parser.add_argument('files', nargs='+', help='im1, labels1, im2, labels2_output files')

    pargs = parser.parse_args(args[1:])

    print(str(inspect.getargspec(cmds[pargs.subcmd]['fn'])))

    # complain if not all four input/output args
    if len(pargs.files) % 4 != 0:
        parser.error( 'must specify input and output image and labels' )

    run_jobs(pargs)

if __name__ == '__main__':
    sys.exit(main(*sys.argv))
