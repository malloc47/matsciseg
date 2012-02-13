#!/usr/bin/env python
import sys,os
import numpy as np
# sys.path.insert(0,os.getcwd() + '/gco')
# sys.path.insert(0,os.getcwd())
# import gcoc,gco
import wx
import cv,cv2
from math import sqrt

jet_colors = [(0,0,143), (0,0,159), (0,0,175), (0,0,191), (0,0,207), (0,0,223), (0,0,239), (0,0,255), (0,16,255), (0,32,255), (0,48,255), (0,64,255), (0,80,255), (0,96,255), (0,112,255), (0,128,255), (0,143,255), (0,159,255), (0,175,255), (0,191,255), (0,207,255), (0,223,255), (0,239,255), (0,255,255), (16,255,239), (32,255,223), (48,255,207), (64,255,191), (80,255,175), (96,255,159), (112,255,143), (128,255,128), (143,255,112), (159,255,96), (175,255,80), (191,255,64), (207,255,48), (223,255,32), (239,255,16), (255,255,0), (255,239,0), (255,223,0), (255,207,0), (255,191,0), (255,175,0), (255,159,0), (255,143,0), (255,128,0), (255,112,0), (255,96,0), (255,80,0), (255,64,0), (255,48,0), (255,32,0), (255,16,0), (255,0,0), (239,0,0), (223,0,0), (207,0,0), (191,0,0), (175,0,0), (159,0,0), (143,0,0), (128,0,0), (0,0,159), (0,0,175), (0,0,191), (0,0,207), (0,0,223), (0,0,239), (0,0,255), (0,16,255), (0,32,255), (0,48,255), (0,64,255), (0,80,255), (0,96,255), (0,112,255), (0,128,255), (0,143,255), (0,159,255), (0,175,255), (0,191,255), (0,207,255), (0,223,255), (0,239,255), (0,255,255), (16,255,239), (32,255,223), (48,255,207), (64,255,191), (80,255,175), (96,255,159), (112,255,143), (128,255,128), (143,255,112), (159,255,96), (175,255,80), (191,255,64), (207,255,48), (223,255,32), (239,255,16), (255,255,0), (255,239,0), (255,223,0), (255,207,0), (255,191,0), (255,175,0), (255,159,0), (255,143,0), (255,128,0), (255,112,0), (255,96,0), (255,80,0), (255,64,0), (255,48,0), (255,32,0), (255,16,0), (255,0,0), (239,0,0), (223,0,0), (207,0,0), (191,0,0), (175,0,0), (159,0,0), (143,0,0), (128,0,0)]
randmap = [54,6,46,17,35,63,1,23,28,27,49,2,13,21,14,39,30,47,45,4,26,5,57,61,55,52,32,34,41,18,12,25,53,56,22,19,48,9,37,8,60,36,64,16,3,58,44,15,7,10,20,24,40,11,43,50,38,51,59,29,62,31,33,42,54,6,46,17,35,63,1,23,28,27,49,2,13,21,14,39,30,47,45,4,26,5,57,61,55,52,32,34,41,18,12,25,53,56,22,19,48,9,37,8,60,36,64,16,3,58,44,15,7,10,20,24,40,11,43,50,38,51,59,29,62,31,33,42]

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

def grey_to_rgb(im):
    return np.repeat(im,3,axis=1).reshape(im.shape+(3,))

def color_jet(img,labels,alpha=0.5):
    output = img.copy()
    for l in range(labels.min(),labels.max()):
        output[labels==(l%128)] = jet_colors[randmap[l%128]]
    output = (output*alpha+img*(1-alpha)).astype('uint8')
    return output

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
    im = cv.LoadImageM("seq1/img/image0091.tif")
    im_gray = cv.CreateMat(im.rows, im.cols, cv.CV_8U)
    cv.CvtColor(im,im_gray, cv.CV_RGB2GRAY)
    # convert to numpy arrays
    im_gray = np.asarray(im_gray[:,:])
    im = np.asarray(im[:,:])
    seed=np.genfromtxt("seq1/labels/image0091.label",dtype='int16')
    w=Window(im_gray,seed)
