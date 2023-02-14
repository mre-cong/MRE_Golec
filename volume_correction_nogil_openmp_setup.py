# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 10:49:07 2023

@author: bagaw
"""

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy
ext_modules = [
      Extension(
            "get_volume_correction_force_cy_nogil",
            ["get_volume_correction_force_cy_nogil.pyx"],
            extra_compile_args=['-fopenmp'],
            extra_link_args = ['-fopenmp'],
      )
]

setup(name='get_volume_correction_force_cy_nogil',
      ext_modules=cythonize(ext_modules,language_level=3),
      include_dirs=[numpy.get_include()]) 