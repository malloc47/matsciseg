# Django settings for webgui project.

import sys,os.path
sys.path.insert(0,os.getcwd() + '/..')
import matsci.gco


DEBUG = True
TEMPLATE_DEBUG = DEBUG

ADMINS = (
    # ('Your Name', 'your_email@example.com'),
)

MANAGERS = ADMINS

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3', # Add 'postgresql_psycopg2', 'mysql', 'sqlite3' or 'oracle'.
        'NAME': 'matsci.db',                      # Or path to database file if using sqlite3.
        'USER': '',                      # Not used with sqlite3.
        'PASSWORD': '',                  # Not used with sqlite3.
        'HOST': '',                      # Set to empty string for localhost. Not used with sqlite3.
        'PORT': '',                      # Set to empty string for default. Not used with sqlite3.
    }
}

# Local time zone for this installation. Choices can be found here:
# http://en.wikipedia.org/wiki/List_of_tz_zones_by_name
# although not all choices may be available on all operating systems.
# On Unix systems, a value of None will cause Django to use the same
# timezone as the operating system.
# If running in a Windows environment this must be set to the same as your
# system time zone.
TIME_ZONE = 'America/Chicago'

# Language code for this installation. All choices can be found here:
# http://www.i18nguy.com/unicode/language-identifiers.html
LANGUAGE_CODE = 'en-us'

SITE_ID = 1

# If you set this to False, Django will make some optimizations so as not
# to load the internationalization machinery.
USE_I18N = True

# If you set this to False, Django will not format dates, numbers and
# calendars according to the current locale.
USE_L10N = True

# If you set this to False, Django will not use timezone-aware datetimes.
USE_TZ = True

# Absolute filesystem path to the directory that will hold user-uploaded files.
# Example: "/home/media/media.lawrence.com/media/"
MEDIA_ROOT = ''

# URL that handles the media served from MEDIA_ROOT. Make sure to use a
# trailing slash.
# Examples: "http://media.lawrence.com/media/", "http://example.com/media/"
MEDIA_URL = ''

# Absolute path to the directory static files should be collected to.
# Don't put anything in this directory yourself; store your static files
# in apps' "static/" subdirectories and in STATICFILES_DIRS.
# Example: "/home/media/media.lawrence.com/static/"
STATIC_ROOT = ''

# URL prefix for static files.
# Example: "http://media.lawrence.com/static/"
STATIC_URL = '/static/'

# Additional locations of static files
STATICFILES_DIRS = (
    # Put strings here, like "/home/html/static" or "C:/www/django/static".
    # Always use forward slashes, even on Windows.
    # Don't forget to use absolute paths, not relative paths.
)

# List of finder classes that know how to find static files in
# various locations.
STATICFILES_FINDERS = (
    'django.contrib.staticfiles.finders.FileSystemFinder',
    'django.contrib.staticfiles.finders.AppDirectoriesFinder',
#    'django.contrib.staticfiles.finders.DefaultStorageFinder',
)

# Make this unique, and don't share it with anybody.
SECRET_KEY = '&amp;k-pft1gtgg&amp;i!cth*2_gybako)aqr1h55)3hbi)zn^d6m$v26'

# List of callables that know how to import templates from various sources.
TEMPLATE_LOADERS = (
    'django.template.loaders.filesystem.Loader',
    'django.template.loaders.app_directories.Loader',
#     'django.template.loaders.eggs.Loader',
)

MIDDLEWARE_CLASSES = (
    'django.middleware.common.CommonMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    # Uncomment the next line for simple clickjacking protection:
    # 'django.middleware.clickjacking.XFrameOptionsMiddleware',
)

ROOT_URLCONF = 'webgui.urls'

# Python dotted path to the WSGI application used by Django's runserver.
WSGI_APPLICATION = 'webgui.wsgi.application'

TEMPLATE_DIRS = (
    # Put strings here, like "/home/html/django_templates" or "C:/www/django/templates".
    # Always use forward slashes, even on Windows.
    # Don't forget to use absolute paths, not relative paths.
    os.path.join(os.path.dirname(__file__), 'templates'),
)

CACHES = {
    'default': {
        'BACKEND': 'common.cache.LocMemRawCache',
        'TIMEOUT': 1200,
        'OPTIONS': {
            'MAX_ENTRIES' : 30,
            'CULL_FREQUENCY' : 4,
        },
    }
}

INSTALLED_APPS = (
    # 'django.contrib.auth',
    # 'django.contrib.contenttypes',
    # 'django.contrib.sessions',
    # 'django.contrib.sites',
    # 'django.contrib.messages',
    # 'django.contrib.staticfiles',
    # Uncomment the next line to enable the admin:
    # 'django.contrib.admin',
    # Uncomment the next line to enable admin documentation:
    # 'django.contrib.admindocs',
)

# A sample logging configuration. The only tangible logging
# performed by this configuration is to send an email to
# the site admins on every HTTP 500 error when DEBUG=False.
# See http://docs.djangoproject.com/en/dev/topics/logging for
# more details on how to customize your logging configuration.
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'filters': {
        'require_debug_false': {
            '()': 'django.utils.log.RequireDebugFalse'
        }
    },
    'handlers': {
        'mail_admins': {
            'level': 'ERROR',
            'filters': ['require_debug_false'],
            'class': 'django.utils.log.AdminEmailHandler'
        }
    },
    'loggers': {
        'django.request': {
            'handlers': ['mail_admins'],
            'level': 'ERROR',
            'propagate': True,
        },
    }
}

from glob import glob

DATA_SOURCE = 'data/'

datasets = { path : 
             # { int(os.path.splitext(os.path.basename(f))[0]) : f  
             { i : f  
               for i,f in zip( range(len(glob(os.path.join(DATA_SOURCE,path) + '/*.npz'))),
                             sorted(glob(os.path.join(DATA_SOURCE,path) + '/*.npz'))) }
             for path in [ d for d in os.listdir(DATA_SOURCE) 
                           if os.path.isdir(os.path.join(DATA_SOURCE, d)) ] }

import cPickle as pickle
print('loading dataset')
slices = pickle.load(open('c2a.pkl','rb'))
current_img = min(slices.keys())
print('done loading dataset')

#import cPickle as pickle
#from numpy import genfromtxt
#current_img=91
#images = range(90,96)
#slices = {}
#for i in images:
	#print(str(i))
	##im,im_gray = matsciskel.read_img('../seq1/img/image'+format(i,'04d')+'.png')
	#im,im_gray = matsciskel.read_img('../seq1/cropped3/image'+format(i,'04d')+'.png')
	##seed = pickle.load(open(format(i,'04d')+'.pkl','rb'))
	## seed=genfromtxt('../seq1/global-20/90/image'+format(i,'04d')+'.label',dtype='int16')
	#seed=genfromtxt('../seq1/cropped3/image'+format(i,'04d')+'.label',dtype='int16')
	## pickle.dump(seed,open(format(i,'04d')+'.pkl','wb'))
	#v = matsci.gco.Volume(im_gray,seed)
	#slices[i] = v
#pickle.dump(slices,open('c3.pkl','wb'))
