#!/usr/bin/env python
from __future__ import print_function
import sys,os,cv,cv2
import numpy as np
import scipy
from scipy import ndimage
import recipes
import argparse
import inspect
import matsciskel

edge_types = {
    'i' : 0,
    'e' : 1,
    'm' : 2,
    's' : 3,
    't' : 4,
    }

def run_jobs(pargs,fn):
    for f in range(0,len(pargs.files),4):        
        im_prev,im_prev_gray =  matsciskel.read_img(pargs.files[f])
        im,im_gray =  matsciskel.read_img(pargs.files[f+2])
        if pargs.files[f+1].endswith(('.png','.jpg')):
            from matsci.label import pts_to_label
            labels_prev = pts_to_label(scipy.misc.imread(pargs.files[f+1],flatten=True).astype('bool'))
        else:
            labels_prev =  np.genfromtxt(pargs.files[f+1],dtype='int16')
        labels_out = pargs.files[f+3]
        param = {
            'im' : im,
            'im_gray' : im_gray,
            'im_prev' : im_prev,
            'im_prev_gray' : im_prev_gray,
            'labels' :  labels_prev,
            'labels_out' : labels_out,
            'vp' : print if pargs.verbose else lambda *a, **k: None
            }
        param.update(vars(pargs))
        output = fn(**{k:v for k,v in param.iteritems() if k in inspect.getargspec(fn).args})
        np.savetxt(labels_out,output,fmt='%1d')
        if not pargs.output_image is None:
            cv2.imwrite(pargs.output_image,
                        matsciskel.draw_on_img(im,
                                               matsciskel.label_to_bmp(output)))

def main(*args):
    parser = argparse.ArgumentParser(
        description='CLI for materials image segmentation propagation')
    subparsers = parser.add_subparsers(dest='subcmd')

    parser.add_argument('-v', '--verbose', action='count',
                        default=0, dest='verbose')

    cmds = recipes.cmds

    for cmdname, cmdval in cmds.iteritems():
        cmd_parser = subparsers.add_parser(cmdname)
        for argname, argval in cmdval['args'].iteritems():
            # the only "blessed" parameter is the name field
            namelst = argval['name'] if 'name' in argval else ['--'+argname]
            cmd_parser.add_argument(
                *namelst,
                 **{k:v for k,v in argval.iteritems() if k!='name'})

        cmd_parser.add_argument('-o','--output-image',
                                action='store',dest='output_image',
                                help='output image path to store visualization of segmentation')
        cmd_parser.add_argument('files', nargs='+', help='im_prev, labels_prev, im, labels_output files')

    pargs = parser.parse_args(args[1:])

    # complain if not given all four input/output args
    if len(pargs.files) % 4 != 0:
        parser.error( 'must specify input and output image and labels' )

    run_jobs(pargs, cmds[pargs.subcmd]['fn'])

if __name__ == '__main__':
    sys.exit(main(*sys.argv))
