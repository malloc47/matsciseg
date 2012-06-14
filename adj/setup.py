#-------------------------------------------------------------------------------
from distutils.core import setup, Extension;
import numpy as np;
#-------------------------------------------------------------------------------
setup(name="adjc",
      ext_modules=[Extension("adjc",sources=["adjc.cpp","RegionGraph.cpp"],
                             include_dirs=[np.get_include()],
                             extra_compile_args=["-fpermissive"])]);
#-------------------------------------------------------------------------------
