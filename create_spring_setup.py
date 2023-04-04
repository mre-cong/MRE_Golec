from distutils.core import setup
from Cython.Build import cythonize
import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True
import numpy


setup(name='create_springs',
      ext_modules=cythonize("create_springs.pyx",language_level=3, annotate=True),
      include_dirs=[numpy.get_include()])
# setup(name='get_spring_force_cy',
#       ext_modules=cythonize("get_spring_force_cy.pyx",language_level=3),
#       include_dirs=[numpy.get_include()]) 