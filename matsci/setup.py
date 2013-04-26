from distutils.core import setup, Extension
# from distutils.sysconfig import get_python_inc
import numpy as np
setup(name = "gcoc",
      version = "1.0",
      ext_modules = [Extension("gcoc", ["gcoc.cpp"],
                               include_dirs=[np.get_include(),"../gco"],
                               library_dirs=['../gco'],
                               libraries=['gco'],
                               language='c++')])
