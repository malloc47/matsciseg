from distutils.core import setup, Extension
# from distutils.sysconfig import get_python_inc
# import numpy as np
import numpy.distutils.misc_util
setup(name = "topofix",
      version = "1.0",
      ext_modules = [Extension("topofix", ["topofix.cpp"],
                               include_dirs=numpy.distutils.misc_util.get_numpy_include_dirs(),
                               # include_dirs=[np.get_include(), ],
                               language='c++')])
