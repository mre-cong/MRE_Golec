from distutils.core import setup
from Cython.Build import cythonize

setup(name='update_positions_cy',
      ext_modules=cythonize("update_positions_cy_nogil.pyx",language_level=3)) 