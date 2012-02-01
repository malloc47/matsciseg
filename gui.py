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

def color_jet(img,labels,alpha=0.5):
    colors = [(0,0,143), (0,0,159), (0,0,175), (0,0,191), (0,0,207), (0,0,223), (0,0,239), (0,0,255), (0,16,255), (0,32,255), (0,48,255), (0,64,255), (0,80,255), (0,96,255), (0,112,255), (0,128,255), (0,143,255), (0,159,255), (0,175,255), (0,191,255), (0,207,255), (0,223,255), (0,239,255), (0,255,255), (16,255,239), (32,255,223), (48,255,207), (64,255,191), (80,255,175), (96,255,159), (112,255,143), (128,255,128), (143,255,112), (159,255,96), (175,255,80), (191,255,64), (207,255,48), (223,255,32), (239,255,16), (255,255,0), (255,239,0), (255,223,0), (255,207,0), (255,191,0), (255,175,0), (255,159,0), (255,143,0), (255,128,0), (255,112,0), (255,96,0), (255,80,0), (255,64,0), (255,48,0), (255,32,0), (255,16,0), (255,0,0), (239,0,0), (223,0,0), (207,0,0), (191,0,0), (175,0,0), (159,0,0), (143,0,0), (128,0,0), (0,0,159), (0,0,175), (0,0,191), (0,0,207), (0,0,223), (0,0,239), (0,0,255), (0,16,255), (0,32,255), (0,48,255), (0,64,255), (0,80,255), (0,96,255), (0,112,255), (0,128,255), (0,143,255), (0,159,255), (0,175,255), (0,191,255), (0,207,255), (0,223,255), (0,239,255), (0,255,255), (16,255,239), (32,255,223), (48,255,207), (64,255,191), (80,255,175), (96,255,159), (112,255,143), (128,255,128), (143,255,112), (159,255,96), (175,255,80), (191,255,64), (207,255,48), (223,255,32), (239,255,16), (255,255,0), (255,239,0), (255,223,0), (255,207,0), (255,191,0), (255,175,0), (255,159,0), (255,143,0), (255,128,0), (255,112,0), (255,96,0), (255,80,0), (255,64,0), (255,48,0), (255,32,0), (255,16,0), (255,0,0), (239,0,0), (223,0,0), (207,0,0), (191,0,0), (175,0,0), (159,0,0), (143,0,0), (128,0,0)]
    randmap = [54,6,46,17,35,63,1,23,28,27,49,2,13,21,14,39,30,47,45,4,26,5,57,61,55,52,32,34,41,18,12,25,53,56,22,19,48,9,37,8,60,36,64,16,3,58,44,15,7,10,20,24,40,11,43,50,38,51,59,29,62,31,33,42,54,6,46,17,35,63,1,23,28,27,49,2,13,21,14,39,30,47,45,4,26,5,57,61,55,52,32,34,41,18,12,25,53,56,22,19,48,9,37,8,60,36,64,16,3,58,44,15,7,10,20,24,40,11,43,50,38,51,59,29,62,31,33,42]
    output = img.copy()
    for l in range(labels.min(),labels.max()):
        output[labels==(l%128)] = colors[randmap[l%128]]
    output = (output*alpha+img*(1-alpha)).astype('uint8')
    return output

class Window():
    def lclick(self,event):
        if(event.GetPosition()[1] < self.img.shape[0] and 
           event.GetPosition()[0] < self.img.shape[1]):
            # self.img[event.GetPosition()[1],event.GetPosition()[0],:] = (255,0,0)
            self.add_label(event.GetPosition())
            self.redraw_img()
            # print(event.GetPosition())

    def rclick(self,event):
        if(event.GetPosition()[1] < self.img.shape[0] and 
           event.GetPosition()[0] < self.img.shape[1]):
            print(event.GetPosition())

    def redraw_img(self):
        self.bmp.SetBitmap(numpy_to_bitmap(self.img))
        # self.bmp.SetBitmap(numpy_to_bitmap(color_jet(self.img_orig,self.labels)))

    def recreate_img(self):
        self.img = self.background.copy()
        for p in self.create_list:
            self.img[p[1],p[0],:] = (255,0,0)

    def add_label(self,p):
        q = nearest_neighbor(p,self.create_list,5)
        if q is not None:
            print(q)
            self.create_list = filter(lambda x: x!=q , self.create_list)
        else:
            self.create_list.append(tuple(p)+(1,))
            # self.img[p[1],p[0],:] = (255,0,0)
        self.recreate_img()
        print(self.create_list)

    def __init__(self,img,seed):
        # setup local attributes
        self.create_list = []
        self.remove_list = []
        # create main window, frame, and panel
        app = wx.PySimpleApp()
        frame = wx.Frame(None, -1, 'Segmentation Refinement')
        panel = wx.Panel(frame)
        # setup bitmap control and callbacks
        self.labels = seed.copy()
        self.img = color_jet(img,self.labels)
        self.background = self.img.copy()
        self.img_orig = img.copy()
        bitmap = numpy_to_bitmap(self.img)
        self.bmp = wx.StaticBitmap(panel, -1, bitmap)
        self.bmp.SetPosition((0, 0))
        self.bmp.Bind(wx.EVT_LEFT_DOWN, self.lclick)
        self.bmp.Bind(wx.EVT_RIGHT_DOWN, self.rclick)
        # start the show
        frame.Show()
        app.MainLoop()

def test():
    im = cv.LoadImageM("seq1/img/image0091.tif")
    im_gray = cv.CreateMat(im.rows, im.cols, cv.CV_8U)
    cv.CvtColor(im,im_gray, cv.CV_RGB2GRAY)
    # convert to numpy arrays
    im_gray = np.asarray(im_gray[:,:])
    im = np.asarray(im[:,:])
    seed=np.genfromtxt("seq1/labels/image0091.label",dtype='int16')
    w=Window(im,seed)
