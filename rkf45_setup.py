from distutils.core import setup
from Cython.Build import cythonize

setup(name='rkf45',
      ext_modules=cythonize("rkf45.pyx",language_level=3)) 