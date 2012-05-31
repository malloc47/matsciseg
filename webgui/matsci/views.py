# Create your views here.
from django.http import HttpResponse, HttpResponseBadRequest
from django.template import Context, loader
from django.shortcuts import render_to_response
import simplejson as json
from webgui.settings import current_img, images, slices

import scipy.ndimage.interpolation
import scipy.misc
from PIL import Image
import numpy as np
import gui

def index(req):
    # t = loader.get_template('matsci.html')
    # c = Context({})
    return render_to_response('matsci.html', {})

# def cmd(req):
#     return render_to_response('matsci.html', {})

def img_thumb(request,imgnum):
    slicenum=int(imgnum)
    # image_data = open("/home/malloc47/src/projects/matsci/matsciskel/webgui/img/thumb/image"+imgnum+".png", "rb").read()
    if slicenum in slices:
        # output = scipy.ndimage.interpolation.zoom(slices[slicenum].img,0.15)
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
        output = gui.color_jet(gui.grey_to_rgb(slices[slicenum].img),slices[slicenum].labels)
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

def cmd(request):
    if not request.is_ajax():
        return HttpResponseBadRequest()

    if request.method == 'GET':
        data = handlers[request.GET['method']](request.GET)
        return HttpResponse(json.dumps(data),
                            content_type='application/javascript; charset=utf8')
    return HttpResponseBadRequest()

def handle_addition(params):
    print('Addition')
    return 'success'

def handle_removal(params):
    print('Removal')
    return 'success'

def handle_global(params):
    print('Global')
    return 'global graph cut successful'

def handle_copyr(params):
    print('copyr')
    return 'copyr successful'

def handle_copyl(params):
    print('copyl')
    return 'copyl successful'

def handle_revert(params):
    print('revert')
    return 'revert successful'

def handle_reload(params):
    import matsciskel,pickle,gco
    global current_img, images, slices
    current_img=91
    images = range(90,92)
    slices = {}
    for i in images:
        print(str(i))
        im,im_gray = matsciskel.read_img('../seq1/img/image'+format(i,'04d')+'.png')
        seed = pickle.load(open(format(i,'04d')+'.pkl','rb'))
        v = gco.Volume(im_gray,seed)
        slices[i] = v
    return 'reload successful'

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
    slices[int(current_img)].edit_labels(5,addition,removal)
    return 'local graph cut successful'

def handle_click(params):
    print('Click')
    return 'x:'+params['x']+',y:'+params['y']

def load(request):
    global current_img, images
    l = [current_img]+images
    return HttpResponse(json.dumps(l),
                        content_type='application/javascript; charset=utf8')

def change_img(request):
    global current_img, images
    print(response.GET['img'])
    if not response.GET['img'] in images:
        return HttpResponseBadRequest()
    current_img = response.GET['img']
    return HttpResponse(json.dumps({'response':'success'}),
                        content_type='application/javascript; charset=utf8')

def state(request):
    global current_img, images, slices
    data = None;
    if('image' in request.GET):
        current_img = request.GET['image'];
    if('images' in request.GET):
        images = request.GET['images'];
    if('command' in request.GET):
        handlers = {
            'addition' : handle_addition,
            'removal'  : handle_removal,
            'global'   : handle_global,
            'local'    : handle_local,
            # 'imgclick' : handle_click,
            'copyr'    : handle_copyr,
            'copyl'    : handle_copyl,
            'revert'    : handle_revert,
            'reload'    : handle_reload,
            }
        data = handlers[request.GET['command']](request.GET)

        slices[int(current_img)].img.shape[0]

    state_output = {
        'image' : current_img,
        'images': images,
        'height': slices[int(current_img)].img.shape[0],
        'width' : slices[int(current_img)].img.shape[1],
        }

    if not data is None:
        state_output['response'] = data;

    return HttpResponse(json.dumps(state_output),
                        content_type='application/javascript; charset=utf8')
