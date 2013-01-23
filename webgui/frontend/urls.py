from django.conf.urls.defaults import patterns, url

urlpatterns = patterns('frontend.views',
    url(r'^$', 'index'),
    url(r'^cmd/$', 'cmd'),
    url(r'^state/$', 'state'),
    url(r'^datasets/$', 'datasets'),
    url(r'^thumb/(?P<dataset>\w+)/(?P<imgnum>\d{4})/$', 'img_thumb'),
    url(r'^output/(?P<dataset>\w+)/(?P<imgnum>\d{4})/$', 'img_full'),
    url(r'^empty/(?P<dataset>\w+)/(?P<imgnum>\d{4})/$', 'img_full_bare'),
    url(r'^edge/(?P<dataset>\w+)/(?P<imgnum>\d{4})/$', 'img_full_edge'),
)
