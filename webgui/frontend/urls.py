from django.conf.urls.defaults import patterns, url

urlpatterns = patterns('frontend.views',
    url(r'^$', 'index'),
    url(r'^cmd/$', 'cmd'),
    url(r'^state/$', 'state'),
    url(r'^datasets/$', 'get_datasets'),
    url(r'^dataset/$', 'load_dataset'),
    url(r'^raw/$', 'img_raw'),
    url(r'^labeled/$', 'img_labeled'),
    url(r'^edge/$', 'img_edge'),
    url(r'^thumb/$', 'img_thumb'),
    url(r'^copy/$', 'copy'),
    url(r'^global/$', 'globalgc'),
    url(r'^local/$', 'localgc'),
    url(r'^reset/$', 'reset'),
)
