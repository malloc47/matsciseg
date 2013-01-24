from django.http import HttpResponse, HttpResponseBadRequest
from django.template import Context, loader
from django.shortcuts import render_to_response
from django.core.cache import cache
import simplejson as json

import scipy.ndimage.interpolation
import scipy.misc
import numpy as np
from PIL import Image

from webgui.settings import datasets

import render_labels
import matsci.gui
import matsci

def fetch_or_load(dataset,index):
    # avoid using default value since it would be evaluated
    v = cache.get(dataset+'_'+index)
    if v is None:
        print(str("Reloading from cache"))
        v = matsci.gco.Slice.load(datasets[dataset][int(index)])
        cache.set(dataset+'_'+index , v)
    return v

def retrieve_cached(fn):
    def wrap(request):
        if request.method != 'GET':
            return HttpResponseBadRequest()
        try:
            v = fetch_or_load(request.GET['dataset'], request.GET['slice'])
        except:
            return HttpResponseBadRequest()
        return fn(request,v)
    return wrap

def retrieve_two_cached(fn):
    def wrap(request):
        if request.method != 'GET':
            return HttpResponseBadRequest()
        try:
            v = fetch_or_load(request.GET['dataset'], request.GET['slice'])
            u = fetch_or_load(request.GET['dataset'], request.GET['source'])
        except:
            return HttpResponseBadRequest()
        return fn(request,v,u)
    return wrap

@retrieve_cached
def img_raw(request,v):
    http_output = Image.fromarray(np.uint8(v.img))
    response = HttpResponse(mimetype="image/png")
    http_output.save(response, "PNG")
    return response

@retrieve_cached
def img_labeled(request,v):
    output = matsci.gui.color_jet(matsci.gui.grey_to_rgb(v.img),v.labels.v)
    http_output = Image.fromarray(np.uint8(output))
    response = HttpResponse(mimetype="image/png")
    http_output.save(response, "PNG")
    return response

@retrieve_cached
def img_thumb(request,v):
    http_output = Image.fromarray(np.uint8(v.img))
    http_output.thumbnail((128,128), Image.ANTIALIAS)
    response = HttpResponse(mimetype="image/png")
    http_output.save(response, "PNG")
    return response

@retrieve_cached
def img_edge(request,v):
    im = v.img
    im = np.dstack((im,im,im))
    bmp = render_labels.label_to_bmp(v.labels.v)
    output = render_labels.draw_on_img(im,bmp)
    http_output = Image.fromarray(np.uint8(output))
    response = HttpResponse(mimetype="image/png")
    http_output.save(response, "PNG")
    return response

def index(req):
    return render_to_response('matsci.html', {})

def load_dataset(request):
    if not request.is_ajax():
        return HttpResponseBadRequest()
    if request.method == 'GET':
        try:
            d = range(len(datasets[request.GET['dataset']]))
            return HttpResponse(json.dumps(d),
                                content_type='application/javascript; charset=utf8')
        except:
            return HttpResponseBadRequest()
    return HttpResponseBadRequest()

def get_datasets(request):
    if not request.is_ajax():
        return HttpResponseBadRequest()

    if request.method == 'GET':
        labels = {
            'ti' : 'Ti-26', 
            'c1' : 'Crop1', 
            'c2' : 'Crop2', 
            'c3' : 'Crop3',
            'c4' : 'Crop4'
            }
        d = [ [f,labels[f] ] for f in datasets.keys() ]

        return HttpResponse(json.dumps(d),
                            content_type='application/javascript; charset=utf8')
    return HttpResponseBadRequest()

@retrieve_two_cached
def copy(request,v,u):
    cache.set(request.GET['dataset']+'_'+request.GET['slice'] , 
              matsci.gco.Slice(v.img,u.labels.v))
    return HttpResponse(json.dumps('copyr successful'),
                        content_type='application/javascript; charset=utf8')

@retrieve_cached
def globalgc(request,v):
    dilation = int(request.GET['dilation'])
    # the local operation doesn't regenerate the data term over the
    # whole image to speed interaction (positing global is rare)
    # if needed, regenerate here
    try:
        v.data.dilate_all(dilation)
    except AttributeError:
        v.data = matsci.data.Data(v.labels.v)
        v.data.dilate_all(dilation)
        
    v.graph_cut(1)
    return HttpResponse(json.dumps('global graph cut successful'),
                        content_type='application/javascript; charset=utf8')

def convert_string(s):
    return [tuple(map(int,i.split(','))) 
            for i in s.split(';')]

@retrieve_cached
def localgc(request,v):
    params = request.GET
    global current_img, images, slices
    print('Local')
    addition = [];
    auto = [];
    removal = [];
    line = [];
    if 'addition' in params and params['addition']:
        addition = convert_string(params['addition'])
    if 'auto' in params and params['auto']:
        auto = convert_string(params['auto'])
    if 'removal' in params and params['removal']:
        removal = convert_string(params['removal'])
    if 'line' in params and params['line']:
        line = convert_string(params['line'])
    size = int(params['size'])
    dilation = int(params['dilation'])
    # x,y to i,j
    addition = [(b,a,size,dilation) for a,b in addition]
    auto = [(b,a,size,dilation) for a,b in auto]
    removal = [(b,a) for a,b in removal]
    line = [(b,a,d,c,size,dilation) for a,b,c,d in 
            [l for l in line if len(l)==4]]
    v.edit_labels(addition,auto,removal,line)
    return HttpResponse(json.dumps('local graph cut successful'),
                        content_type='application/javascript; charset=utf8')
