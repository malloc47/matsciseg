#!/usr/bin/env python
import sys,os,cv,cv2
import numpy as np
sys.path.insert(0,os.getcwd() + '/gco')
sys.path.insert(0,os.getcwd())
import gcoc,gco
import scipy
from scipy import ndimage

def display(im):
    cv2.namedWindow("tmp",cv2.CV_WINDOW_AUTOSIZE)
    cv2.imshow("tmp",im)
    cv2.waitKey(0)

def label_to_bmp(labels):
    grad = np.gradient(labels)
    seg = np.maximum(abs(grad[0]),abs(grad[1]))
    return seg

def draw_on_img(img,bmp):
    out = img.copy()
    out[np.nonzero(bmp>0)] = (255,0,0)
    return out

def read_img(img_name):
    im = cv.LoadImageM(img_name)
    im_gray = cv.CreateMat(im.rows, im.cols, cv.CV_8U)
    cv.CvtColor(im,im_gray, cv.CV_RGB2GRAY)
    # convert to numpy arrays
    im_gray = np.asarray(im_gray[:,:])
    im = np.asarray(im[:,:])
    return (im,im_gray)

def imgio(fn):
    def imghandle(arg):
        im,im_gray = read_img(arg['im2'])
        seed=np.genfromtxt(arg['label'],dtype='int16')
        output = fn(arg,im,im_gray,seed)
        np.savetxt(arg['label_out'],output,fmt='%1d')
        # bmp_labels = label_to_bmp(output)
        # cv2.imwrite(arg['im_out'],draw_on_img(im,bmp_labels))
        cv2.imwrite(arg['im_out'],draw_on_img(im,label_to_bmp(output)))
    return imghandle

@imgio
def global_process(arg,im,im_gray,seed):
    v = gco.Volume(im_gray,seed)
    print("Initialized")
    v.dilate_all(arg['d'])
    v.output_data_term()
    print("Dilated")
    # v.set_adj_label_all(0);
    # print("Adjacent")
    v.graph_cut(arg['gctype'])
    print("Graph Cut Complete")
    # import code; code.interact(local=locals())
    return v.labels

@imgio
def global_auto_process(arg,im,im_gray,seed):
    v = gco.Volume(im_gray,seed)
    print("Initialized")
    v.dilate_auto(arg['d'])
    v.output_data_term()
    print("Dilated")
    v.graph_cut(arg['gctype'])
    print("Graph Cut Complete")
    return v.labels

@imgio
def global_interface_process(arg,im,im_gray,seed):
    v = gco.Volume(im_gray,seed)
    print("Initialized")
    v.dilate_all(arg['d'])
    print("Dilated")
    v.graph_cut(arg['gctype'])
    v.edit_labels(5)
    # import code; code.interact(local=locals())
    return v.labels

@imgio
def skel_process(arg,im,im_gray,seed):
    v = gco.Volume(im_gray,seed)
    print("Initialized")
    v.dilate(arg['d'])
    v.dilate_first(arg['d'])
    print("Dilated")
    v.skel()
    return v.graph_cut(arg['gctype'])

@imgio
def gauss_process(arg,im,im_gray,seed):
    v = gco.Volume(im_gray,seed)
    print("Initialized")
    # v.dilate_first(arg['d']/10)
    v.fit_gaussian(arg['d'],arg['d2'],arg['d3'])
    return v.graph_cut(arg['gctype'])

def main(*args):
    if(len(args) < 8):
        return 1

    proc_type = args[1]
    proc = args[2]

    if(args[1] == "e"):
        proc_type = 1
    elif(args[1] == "i"):
        proc_type = 0
    elif(args[1] == "m"):
        proc_type = 2
    else:
        print("Bad argument type")
        return 0

    arg = {'gctype'    : proc_type,
           'im1'       : args[3],
           'label'     : args[4],
           'im2'       : args[5],
           'label_out' : args[6],
           'im_out'    : args[7]}

    procs = {'skel'   : skel_process,
             'global' : global_process,
             'auto' : global_auto_process,
             'guiglobal' : global_interface_process,
             'gauss' : gauss_process}

    if(len(args) > 8):
        arg['d'] = int(args[8]);
        print("Dilating:  "+str(arg['d']))
    if(len(args) > 9):
        arg['d2'] = int(args[9]);
        print("Dilating2: "+str(arg['d2']))
    if(len(args) > 10):
        arg['d3'] = float(args[10]);
        print("Dilating3: "+str(arg['d3']))

    procs[proc](arg)

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
          'seq1/img/image0090.tif',
          'seq1/labels/image0090.label',
          'seq1/img/image0091.tif',
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
