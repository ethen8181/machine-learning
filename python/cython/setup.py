# compiling the .pyx module
import numpy as np
from distutils.core import setup
from Cython.Build import cythonize

# key-value pairs that tell disutils the name
# of the application and which extensions it
# needs to build 
# 1. for the cython modules, we're using glob patterns
# e.g. *.pyx for every .pyx file or simply pass in
# the filename.pyx
# 2. include_dirs, makes sure we can include numpy
setup(
	name = 'Hello',
	ext_modules = cythonize('*.pyx'),
	include_dirs = [np.get_include()]
)

# after that run 
# python setup.py build_ext --inplace
# in the command line, and we can import it like
# normal python modules
