# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 10:49:07 2023

@author: bagaw
"""

from distutils.core import setup
from Cython.Build import cythonize

setup(name='get_volume_correction_force_cy',
      ext_modules=cythonize("get_volume_correction_force_cy.pyx",language_level=3)) 