from django.conf.urls.defaults import patterns, url

urlpatterns = patterns('matsci.views',
    url(r'^$', 'index'),
    url(r'^cmd/$', 'cmd'),
    url(r'^state/$', 'state'),
    url(r'^thumb/(\d{4})/$', 'img_thumb'),
    url(r'^output/(\d{4})/$', 'img_full'),
    url(r'^empty/(\d{4})/$', 'img_full_bare'),
    url(r'^edge/(\d{4})/$', 'img_full_edge'),
)
