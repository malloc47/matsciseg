from django.conf.urls.defaults import patterns, url

urlpatterns = patterns('matsci.views',
    url(r'^$', 'index'),
    url(r'^cmd/$', 'cmd'),
)
