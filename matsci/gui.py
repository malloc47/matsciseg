#!/usr/bin/env python
import sys,os
import numpy as np
# sys.path.insert(0,os.getcwd() + '/gco')
# sys.path.insert(0,os.getcwd())
# import gcoc,gco
import wx
import cv,cv2
from math import sqrt
from draw import color_jet, grey_to_rgb


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

def draw_circle(im,p):
    r = p[2]
    cx, cy = p[0:2]
    y, x = np.ogrid[-r:r, -r:r]
    index = x**2 + y**2 <= r**2
    # handle border cases
    crop = ((cx-r)-max(cx-r,0), 
            min(cx+r,im.shape[0])-(cx+r),
            (cy-r)-max(cy-r,0),
            min(cy+r,im.shape[1])-(cy+r))
    im[cx-r-crop[0]:cx+r+crop[1], cy-r-crop[2]:cy+r+crop[3]][
        index[abs(crop[0]):index.shape[0]-abs(crop[1]),
              abs(crop[2]):index.shape[1]-abs(crop[3])],:] = (255,255,255)


class Window():
    def lclick(self,event):
        p = (event.GetPosition()[1],event.GetPosition()[0])
        if(p[0] < self.background.shape[0] and \
           p[1] < self.background.shape[1]):
            self.add_label(p)
            self.redraw_img(self.recreate_img())
            # print(self.remove_list)
            # print(self.create_list)

    def rclick(self,event):
        p = (event.GetPosition()[1],event.GetPosition()[0])
        if(p[0] < self.background.shape[0] and \
           p[1] < self.background.shape[1]):
            print(self.labels[p[0:2]])
            self.remove_label(p)
            self.redraw_img(self.recreate_img())
            # print(self.remove_list)
            # print(self.create_list)

    def redraw_img(self,im):
        self.bmp.SetBitmap(numpy_to_bitmap(im))

    def recreate_img(self):
        output = self.background.copy()
        for p in self.create_list:
            draw_circle(output,p)
        for p in self.remove_list:
            # print((jet_colors[randmap[p%128]],)*3)
            output[self.labels==p,:] = (0,0,0)
        return output

    def remove_label(self,p):
        q = nearest_neighbor(p,self.create_list,5)
        if q is not None:
            self.create_list = filter(lambda x: x!=q , self.create_list)
        else:
            if self.labels[p[0:2]] in self.remove_list:
                self.remove_list.discard(self.labels[p[0:2]])
            else:
                self.remove_list.add(self.labels[p[0:2]])

    def add_label(self,p):
        q = nearest_neighbor(p,self.create_list,5)
        if q is not None:
            self.create_list = filter(lambda x: x!=q , self.create_list)
            self.create_list.append(tuple(p)+(q[2]+1,))
        else:
            self.create_list.append(tuple(p)+(1,))

    def set_image(self,im,labels):
        self.labels = labels.copy()
        self.background = color_jet(grey_to_rgb(im),self.labels)# im.copy()
        # self.img_orig = im.copy()
        return color_jet(grey_to_rgb(im),self.labels)

    def __init__(self,img,seed):
        # setup local attributes
        self.create_list = []
        self.remove_list = set([])
        # create main window, frame, and panel
        app = wx.PySimpleApp()
        frame = wx.Frame(None, -1, 'Segmentation Refinement',size=(img.shape[1],img.shape[0]))
        panel = wx.Panel(frame)
        # setup bitmap control and callbacks
        self.set_image(img,seed)
        self.bmp = wx.StaticBitmap(panel, -1, 
                                   numpy_to_bitmap(self.set_image(img,seed)))
        self.bmp.SetPosition((0, 0))
        self.bmp.Bind(wx.EVT_LEFT_DOWN, self.lclick)
        self.bmp.Bind(wx.EVT_RIGHT_DOWN, self.rclick)
        # start the show
        frame.Show()
        app.MainLoop()
        print("Exiting GUI")

def test():
    im = cv.LoadImageM("seq1/img/image0091.png")
    im_gray = cv.CreateMat(im.rows, im.cols, cv.CV_8U)
    cv.CvtColor(im,im_gray, cv.CV_RGB2GRAY)
    # convert to numpy arrays
    im_gray = np.asarray(im_gray[:,:])
    im = np.asarray(im[:,:])
    seed=np.genfromtxt("seq1/ground/image0091.label",dtype='int16')
    w=Window(im_gray,seed)
