# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 10:49:07 2023

@author: bagaw
"""

from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(name='get_volume_correction_force_cy_nogil',
      ext_modules=cythonize("get_volume_correction_force_cy_nogil.pyx",language_level=3),
      include_dirs=[numpy.get_include()]) 