#!/usr/bin/env python
import sys,os
import numpy as np
# sys.path.insert(0,os.getcwd() + '/gco')
# sys.path.insert(0,os.getcwd())
# import gcoc,gco
import wx

def numpy_to_bitmap(img):
    output = wx.EmptyImage(img.shape[1],img.shape[0])
    output.SetData(img.tostring())
    return output.ConvertToBitmap()

def scale_bitmap(bitmap, width, height):
    image = wx.ImageFromBitmap(bitmap)
    image = image.Scale(width, height, wx.IMAGE_QUALITY_HIGH)
    result = wx.BitmapFromImage(image)
    return result

def click(event):
    print(event.GetPosition())

class Panel(wx.Panel):
    def __init__(self, parent, img):
        super(Panel, self).__init__(parent, -1)
        bitmap = numpy_to_bitmap(img)
        # print(wx.GetSize(parent))
        # bitmap = scale_bitmap(bitmap, 300, 200)
        control = wx.StaticBitmap(self, -1, bitmap)
        control.SetPosition((0, 0))
        control.Bind(wx.EVT_LEFT_DOWN, click)

    def click(self,event):
        pt = event.GetPosition()
        print(pt)

def create_window(img):
    app = wx.PySimpleApp()
    frame = wx.Frame(None, -1, 'Scaled Image')
    panel = Panel(frame, img)
    frame.Show()
    app.MainLoop()
