#!/usr/bin/env python
import sys,os
import numpy as np
# sys.path.insert(0,os.getcwd() + '/gco')
# sys.path.insert(0,os.getcwd())
# import gcoc,gco
import wx
import cv,cv2
from math import sqrt

def numpy_to_bitmap(img):
    output = wx.EmptyImage(img.shape[1],img.shape[0])
    output.SetData(img.tostring())
    return output.ConvertToBitmap()

def scale_bitmap(bitmap, width, height):
    image = wx.ImageFromBitmap(bitmap)
    image = image.Scale(width, height, wx.IMAGE_QUALITY_HIGH)
    result = wx.BitmapFromImage(image)
    return result

def dist(a,b):
    return sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def nearest_neighbor(p,l,t=float("inf")):
    if not l: return None
    q = min(l,key=lambda x: dist(p,x))
    return q if dist(p,q) < t else None
    

class Window():
    def lclick(self,event):
        if(event.GetPosition()[1] < self.img.shape[0] and 
           event.GetPosition()[0] < self.img.shape[1]):
            # self.img[event.GetPosition()[1],event.GetPosition()[0],:] = (255,0,0)
            self.add_label(event.GetPosition())
            self.redraw_img()
            print(event.GetPosition())

    def rclick(self,event):
        if(event.GetPosition()[1] < self.img.shape[0] and 
           event.GetPosition()[0] < self.img.shape[1]):
            print(event.GetPosition())

    def redraw_img(self):
        self.bmp.SetBitmap(numpy_to_bitmap(self.img))

    def add_label(self,p):
        q = nearest_neighbor(p,self.create_list,5)
        if q is not None:
            pass
        else:
            self.create_list.append(p)
            self.img[p[1],p[0],:] = (255,0,0)

    def __init__(self,img,seed):
        # setup local attributes
        self.create_list = []
        self.remove_list = []
        # create main window, frame, and panel
        app = wx.PySimpleApp()
        frame = wx.Frame(None, -1, 'Segmentation Refinement')
        panel = wx.Panel(frame)
        # setup bitmap control and callbacks
        self.img_orig = img
        self.img = img
        bitmap = numpy_to_bitmap(img)
        self.bmp = wx.StaticBitmap(panel, -1, bitmap)
        self.bmp.SetPosition((0, 0))
        self.bmp.Bind(wx.EVT_LEFT_DOWN, self.lclick)
        self.bmp.Bind(wx.EVT_RIGHT_DOWN, self.rclick)
        # start the show
        frame.Show()
        app.MainLoop()

def test():
    im = cv.LoadImageM("seq1/img/stfl91alss1.tif")
    im_gray = cv.CreateMat(im.rows, im.cols, cv.CV_8U)
    cv.CvtColor(im,im_gray, cv.CV_RGB2GRAY)
    # convert to numpy arrays
    im_gray = np.asarray(im_gray[:,:])
    im = np.asarray(im[:,:])
    seed=np.genfromtxt("seq1/labels/image0091.label",dtype='int16')
    w=Window(im,seed)
