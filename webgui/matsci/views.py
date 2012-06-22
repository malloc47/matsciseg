# Create your views here.
from django.http import HttpResponse, HttpResponseBadRequest
from django.template import Context, loader
from django.shortcuts import render_to_response
import simplejson as json
from webgui.settings import current_img, slices

import scipy.ndimage.interpolation
import scipy.misc
from PIL import Image
import numpy as np
import gui
import cPickle as pickle
import gco

def index(req):
    # t = loader.get_template('matsci.html')
    # c = Context({})
    return render_to_response('matsci.html', {})

def img_thumb(request,imgnum):
    slicenum=int(imgnum)
    if slicenum in slices:
        output = scipy.misc.imresize(slices[slicenum].img,0.15)
        http_output = Image.fromarray(np.uint8(output))
        response = HttpResponse(mimetype="image/png")
        http_output.save(response, "PNG")
        return response
    else:
        return HttpResponseBadRequest()

def img_full(request,imgnum):
    slicenum=int(imgnum)
    if slicenum in slices:
        output = gui.color_jet(gui.grey_to_rgb(slices[slicenum].img),slices[slicenum].labels.v)
        http_output = Image.fromarray(np.uint8(output))
        response = HttpResponse(mimetype="image/png")
        http_output.save(response, "PNG")
        return response
    else:
        return HttpResponseBadRequest()

def img_full_bare(request,imgnum):
    slicenum=int(imgnum)
    if slicenum in slices:
        output = slices[slicenum].img
        http_output = Image.fromarray(np.uint8(output))
        response = HttpResponse(mimetype="image/png")
        http_output.save(response, "PNG")
        return response
    else:
        return HttpResponseBadRequest()

def img_full_edge(request,imgnum):
    import render_labels
    slicenum=int(imgnum)
    if slicenum in slices:
        im = slices[slicenum].img
        im = np.dstack((im,im,im))
        bmp = render_labels.label_to_bmp(slices[slicenum].labels.v)
        output = render_labels.draw_on_img(im,bmp)
        http_output = Image.fromarray(np.uint8(output))
        response = HttpResponse(mimetype="image/png")
        http_output.save(response, "PNG")
        return response
    else:
        return HttpResponseBadRequest()

def cmd(request):
    if not request.is_ajax():
        return HttpResponseBadRequest()

    if request.method == 'GET':
        data = handlers[request.GET['method']](request.GET)
        return HttpResponse(json.dumps(data),
                            content_type='application/javascript; charset=utf8')
    return HttpResponseBadRequest()

# def handle_addition(params):
#     print('Addition')
#     return 'success'

# def handle_removal(params):
#     print('Removal')
#     return 'success'

def handle_copyr(params):
    global current_img, images, slices
    l = sorted(slices.keys())
    idx = l.index(current_img)
    if idx == 0:
        return 'error: no slice on left!'
    old_img = slices[current_img].img;
    slices[current_img] = gco.Slice(old_img,slices[current_img-1].labels.v)
    return 'copyr successful'

def handle_copyl(params):
    global current_img, images, slices
    l = sorted(slices.keys())
    idx = l.index(current_img)
    if idx == len(l)-1:
        return 'error: no slice on right!'
    old_img = slices[current_img].img;
    slices[current_img] = gco.Slice(old_img,slices[current_img+1].labels.v)
    return 'copyl successful'

def handle_dataset(params):
    import pickle
    global current_img, slices
    print('opening dataset')
    slices = pickle.load(open(params['dataset']+'.pkl','rb'))
    current_img = min(slices.keys())
    return 'dataset '+params['dataset']+' opened'

def handle_global(params):
    v = slices[current_img];
    slices[current_img] = gco.Slice(v.img,v.labels.v)
    slices[current_img].data.dilate_all(10)
    slices[current_img].graph_cut(1)
    return 'global graph cut successful'

def handle_local(params):
    global current_img, images, slices
    print('Local')
    # print(str(params))
    addition = [];
    removal = [];
    if 'addition' in params and params['addition']:
        addition = [tuple(map(int,s.split(','))) 
                    for s in params['addition'].split(';')]
    if 'removal' in params and params['removal']:
        removal = [tuple(map(int,s.split(','))) 
                    for s in params['removal'].split(';')]
    slices[current_img].edit_labels(5,addition,removal)
    return 'local graph cut successful'

def handle_click(params):
    print('Click')
    return 'x:'+params['x']+',y:'+params['y']

def load(request):
    global current_img, slices 
    l = [current_img]+sorted(slices.keys())
    return HttpResponse(json.dumps(l),
                        content_type='application/javascript; charset=utf8')

def change_img(request):
    global current_img, slices
    print(response.GET['img'])
    if not response.GET['img'] in sorted(slices.keys()):
        return HttpResponseBadRequest()
    current_img = int(response.GET['img'])
    return HttpResponse(json.dumps({'response':'success'}),
                        content_type='application/javascript; charset=utf8')

def state(request):
    global current_img, slices
    data = None;
    if('image' in request.GET):
        current_img = int(request.GET['image']);
    # if('images' in request.GET):
    #     images = request.GET['images'];
    if('command' in request.GET):
        handlers = {
            # 'addition' : handle_addition,
            # 'removal'  : handle_removal,
            'global'   : handle_global,
            'local'    : handle_local,
            # 'imgclick' : handle_click,
            'copyr'    : handle_copyr,
            'copyl'    : handle_copyl,
            'dataset'    : handle_dataset,
            }
        data = handlers[request.GET['command']](request.GET)

        slices[int(current_img)].img.shape[0]

    state_output = {
        'image' : current_img,
        'images': sorted(slices.keys()),
        'height': slices[int(current_img)].img.shape[0],
        'width' : slices[int(current_img)].img.shape[1],
        }

    if not data is None:
        state_output['response'] = data;

    return HttpResponse(json.dumps(state_output),
                        content_type='application/javascript; charset=utf8')
