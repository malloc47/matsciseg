# Create your views here.
from django.http import HttpResponse
from django.template import Context, loader
from django.shortcuts import render_to_response

def index(req):
    # t = loader.get_template('matsci.html')
    # c = Context({})
    return render_to_response('matsci.html', {})

