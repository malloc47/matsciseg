# Create your views here.
from django.http import HttpResponse, HttpResponseBadRequest
from django.template import Context, loader
from django.shortcuts import render_to_response
import simplejson as json
from webgui.settings import current_img, images

def index(req):
    # t = loader.get_template('matsci.html')
    # c = Context({})
    return render_to_response('matsci.html', {})

# def cmd(req):
#     return render_to_response('matsci.html', {})

def img(request,imgnum):
    image_data = open("/home/malloc47/src/projects/matsci/matsciskel/webgui/img/thumb/image"+imgnum+".png", "rb").read()
    return HttpResponse(image_data, mimetype="image/png")

def cmd(request):
    if not request.is_ajax():
        return HttpResponseBadRequest()

    handlers = {
        'addition' : handle_addition,
        'removal'  : handle_removal,
        'global'   : handle_global,
        'local'    : handle_local,
        'imgclick' : handle_click,
        }

    if request.method == 'GET':
        data = handlers[request.GET['method']](request.GET)
        return HttpResponse(json.dumps(data),
                            content_type='application/javascript; charset=utf8')
    return HttpResponseBadRequest()

def handle_addition(params):
    print('Addition')
    return {'response':'success'}

def handle_removal(params):
    print('Removal')
    return {'response':'success'}

def handle_global(params):
    print('Global')
    return {'response':'success'}

def handle_local(params):
    print('Local')
    return {'response':'success'}

def handle_click(params):
    print('Click')
    return {'response':'x:'+params['x']+',y:'+params['y']}

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
