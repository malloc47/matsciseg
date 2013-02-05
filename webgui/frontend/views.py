from django.http import HttpResponse, HttpResponseBadRequest
from django.template import Context, loader
from django.shortcuts import render_to_response
from django.core.cache import get_cache
import simplejson as json

import scipy.ndimage.interpolation
import scipy.misc
import numpy as np
from PIL import Image

from webgui.settings import datasets

import render_labels
import matsci.gui
import matsci

cache = get_cache('default')
thumbnail_cache = get_cache('thumbnails')


def fetch_or_load(dataset,index):
    # avoid using default value since it would be evaluated
    v = cache.get(dataset+'_'+index)
    if v is None:
        print(str("Reloading from cache"))
        v = matsci.gco.Slice.load(datasets[dataset][int(index)])
        cache.set(dataset+'_'+index , v)
    return v

def fetch_or_load_thumbnail(dataset,index):
    thumb = thumbnail_cache.get(dataset+'_'+index)
    if thumb is None:
        print(str("Reloading thumbnail from cache"))
        # this could be pulled from a the 1st layer of cache...
        v = matsci.gco.Slice.load(datasets[dataset][int(index)])
        thumb = Image.fromarray(np.uint8(v.img))
        thumb.thumbnail((128,128), Image.ANTIALIAS)
        thumbnail_cache.set(dataset+'_'+index , thumb,timeout=31536000)
    return thumb

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

def reset_cache(dataset,index):
    print(str("Reloading from cache"))
    v = matsci.gco.Slice.load(datasets[dataset][int(index)])
    cache.set(dataset+'_'+index , v)
    return v

def reset(request):
    try:
        reset_cache(request.GET['dataset'],request.GET['slice'])
        return HttpResponse(json.dumps('reset cache successful'),
                            content_type='application/javascript; charset=utf8')
    except:
        return HttpResponseBadRequest()

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

def img_thumb(request):
    if request.method != 'GET':
        return HttpResponseBadRequest()
    try:
        http_output = fetch_or_load_thumbnail(request.GET['dataset'], request.GET['slice'])
    except:
        return HttpResponseBadRequest()
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
            'c4' : 'Crop4',
            'c3-demo' : 'RemDemo',
            'c2-demo' : 'AddDemo',
            }

        d = [ [f,labels[f],datasets[f].keys() ] for f in datasets.keys() ]

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

@retrieve_cached
def save(request,v):
    output = datasets[request.GET['dataset']][int(request.GET['slice'])]
    # reload data term, in case it wasn't defined
    v.data = matsci.data.Data(v.labels.v)
    v.save(output)
    return HttpResponse(json.dumps('data save successful'),
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
