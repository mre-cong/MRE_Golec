from distutils.core import setup
from Cython.Build import cythonize
import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True
import numpy


setup(name='springs',
      ext_modules=cythonize("springs.pyx",language_level=3, annotate=True),
      include_dirs=[numpy.get_include()])