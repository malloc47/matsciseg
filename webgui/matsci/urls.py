from django.conf.urls.defaults import patterns, url

urlpatterns = patterns('matsci.views',
    url(r'^$', 'index'),
    url(r'^cmd/$', 'cmd'),
    url(r'^load/$', 'load'),
    url(r'^change/$', 'change_img'),
    url(r'^output/(\d{4})/$', 'img'),
)
