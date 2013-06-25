#!/usr/bin/env python
import sys,os,cv,cv2
import numpy as np
import scipy
from scipy import ndimage
import recipes_old
from matsci.draw import label_to_bmp, draw_on_img
from matsci.io import read_img


def imgio(fn):
    """use to decorate recipes with io ops"""
    def imghandle(arg):
        im,im_gray = read_img(arg['im2'])
        im_prev,im_prev_gray = read_img(arg['im1'])
        seed=np.genfromtxt(arg['label'],dtype='int16')
        output = fn(arg,im,im_gray,im_prev,seed)
        np.savetxt(arg['label_out'],output,fmt='%1d')
        # bmp_labels = label_to_bmp(output)
        # cv2.imwrite(arg['im_out'],draw_on_img(im,bmp_labels))
        cv2.imwrite(arg['im_out'],draw_on_img(im,label_to_bmp(output)))
    return imghandle

def get_recipes():
    """pull in functions from the recipes module, ignoring builtins,
    and strip off the _cmd to get a string equivalent, and then build
    dict of functions"""
    return {k : getattr(recipes_old, k+'_cmd') for k in 
            [ s[:-4] for s in 
              filter(lambda s: 
                     not (s.startswith('_') or 
                          s.endswith('_')) 
                     and s.endswith('_cmd'), 
                     dir(recipes_old))]}

def main(*args):
    if(len(args) < 8):
        return 1

    proc_type = args[1]
    proc = args[2]

    fntype = {
        'i' : 0,
        'e' : 1,
        'm' : 2,
        's' : 3,
        't' : 4,
        }

    try:
        proc_type = fntype[args[1]]
    except:
        print("Bad argument type")
        return 0

    # if(args[1] == "e"):
    #     proc_type = 1
    # elif(args[1] == "i"):
    #     proc_type = 0
    # elif(args[1] == "m"):
    #     proc_type = 2
    # else:
    #     print("Bad argument type")
    #     return 0

    arg = {'gctype'    : proc_type,
           'im1'       : args[3],
           'label'     : args[4],
           'im2'       : args[5],
           'label_out' : args[6],
           'im_out'    : args[7]}

    procs = get_recipes()

    if(len(args) > 8):
        arg['d'] = int(args[8]);
        print("Dilating:  "+str(arg['d']))
    if(len(args) > 9):
        arg['d2'] = int(args[9]);
        print("Dilating2: "+str(arg['d2']))
    if(len(args) > 10):
        arg['d3'] = float(args[10]);
        print("Dilating3: "+str(arg['d3']))

    imgio(procs[proc])(arg)

    return 0

def tests():
    main('./matsciskel.py','i','skel',
          'seq3/img/image0040.png',
          'seq3/labels/image0040.label',
          'seq3/img/image0041.png',
          'test1.label',
          'test1.png',
          '10')
    main('./matsciskel.py','e','global',
          'seq1/img/image0090.png',
          'seq1/ground/image0090.label',
          'seq1/img/image0091.png',
          'test2.label',
          'test2.png',
          '10')
    main('./matsciskel.py','i','gauss',
          'seq3/img/image0040.png',
          'seq3/labels/image0040.label',
          'seq3/img/image0041.png',
          'test3.label',
          'test3.png',
          '10')

if __name__ == '__main__':
    sys.exit(main(*sys.argv))
