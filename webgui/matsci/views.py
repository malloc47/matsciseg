# Create your views here.
from django.http import HttpResponse, HttpResponseBadRequest
from django.template import Context, loader
from django.shortcuts import render_to_response
import simplejson as json

def index(req):
    # t = loader.get_template('matsci.html')
    # c = Context({})
    return render_to_response('matsci.html', {})

# def cmd(req):
#     return render_to_response('matsci.html', {})

def cmd(request):
    if not request.is_ajax():
        return HttpResponseBadRequest()

    handlers = {
        'addition' : handle_addition,
        'removal'  : handle_removal,
        'global'   : handle_global,
        'local'    : handle_local,
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
