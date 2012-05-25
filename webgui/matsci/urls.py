from django.conf.urls.defaults import patterns, url

urlpatterns = patterns('matsci.views',
    url(r'^$', 'index'),
    url(r'^cmd/$', 'cmd'),
    url(r'^state/$', 'state'),
    url(r'^thumb/(\d{4})/$', 'img'),
    url(r'^output/(\d{4})/$', 'img'),
)
