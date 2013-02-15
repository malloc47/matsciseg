#!/usr/bin/env python
import sys,os,cv,cv2
import numpy as np
import scipy
from scipy import ndimage
import recipes
import argparse

edge_types = {
    'i' : 0,
    'e' : 1,
    'm' : 2,
    's' : 3,
    't' : 4,
    }

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

        cmd_parser.add_argument('input', nargs='+', help='input, img, output files')

    pargs = parser.parse_args(args[1:])

    print(str(vars(pargs)))

    # complain if not a triple of optional args
    if len(pargs.input) % 3 != 0:
        parser.error( 'must specify input, target image, and output' )

if __name__ == '__main__':
    sys.exit(main(*sys.argv))
