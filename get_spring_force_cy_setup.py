from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(name='get_spring_force_cy',
      ext_modules=cythonize("get_spring_force_cy.pyx",language_level=3),
      include_dirs=[numpy.get_include()]) 