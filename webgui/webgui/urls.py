from django.conf.urls import patterns, include, url
import webgui.settings

# Uncomment the next two lines to enable the admin:
# from django.contrib import admin
# admin.autodiscover()

urlpatterns = patterns('',
    # Examples:
    # url(r'^$', 'webgui.views.home', name='home'),
    # url(r'^webgui/', include('webgui.foo.urls')),

    # Uncomment the admin/doc line below to enable admin documentation:
    # url(r'^admin/doc/', include('django.contrib.admindocs.urls')),

    # Uncomment the next line to enable the admin:
    # url(r'^admin/', include(admin.site.urls)),
#	url(r'^frontend/', include('matsci.urls')),
	url(r'^', include('frontend.urls')),
)

if webgui.settings.DEBUG:
    urlpatterns += patterns('',
        (r'^img/(?P<path>.*)$', 'django.views.static.serve', {'document_root': 'img'}),
        (r'^js/(?P<path>.*)$', 'django.views.static.serve', {'document_root': 'js'}),
        (r'^css/(?P<path>.*)$', 'django.views.static.serve', {'document_root': 'css'}),
    )
