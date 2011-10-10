from distutils.core import setup, Extension
# from distutils.sysconfig import get_python_inc
import numpy as np
setup(name = "gco",
      version = "1.0",
      ext_modules = [Extension("gco", ["gcomodule.cpp"],include_dirs=[np.get_include(),"/home/malloc47/src/programs/gco"])])
