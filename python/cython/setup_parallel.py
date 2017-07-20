# usually the name should only be setup.py
# on the terminal run
# python setup_parallel.py install
import os
import sys
import glob
import numpy as np
from setuptools import Extension, setup
try:
    from Cython.Build import cythonize
    use_cython = True
except ImportError:
    use_cython = False

# top-level information
NAME = 'pairwise3'
VERSION = '0.0.1'
USE_OPENMP = True


def set_gcc(use_openmp):
    """
    Try to find and use GCC on OSX for OpenMP support

    References
    ----------
    https://github.com/maciejkula/glove-python/blob/master/setup.py
    """
    # For macports and homebrew
    patterns = ['/opt/local/bin/gcc-mp-[0-9].[0-9]',
                '/opt/local/bin/gcc-mp-[0-9]',
                '/usr/local/bin/gcc-[0-9].[0-9]',
                '/usr/local/bin/gcc-[0-9]']

    if 'darwin' in sys.platform.lower():
        gcc_binaries = []
        for pattern in patterns:
            gcc_binaries += glob.glob(pattern)

        gcc_binaries.sort()

        if gcc_binaries:
            _, gcc = os.path.split(gcc_binaries[-1])
            os.environ['CC'] = gcc

        else:
            use_openmp = False

    return use_openmp


def define_extensions(use_cython, use_openmp):
    """
    boilerplate to compile the extension the only thing that we need to
    worry about is the modules part, where we define the extension that
    needs to be compiled
    """
    if sys.platform.startswith('win'):
        # compile args from
        # https://msdn.microsoft.com/en-us/library/fwkeyyhe.aspx
        link_args = []
        compile_args = ['/O2', '/openmp']
    else:
        link_args = []
        compile_args = ['-Wno-unused-function', '-Wno-maybe-uninitialized', '-O3', '-ffast-math']
        if use_openmp:
            compile_args.append('-fopenmp')
            link_args.append('-fopenmp')

        if 'anaconda' not in sys.version.lower():
            compile_args.append('-march=native')

    # recommended approach is that the user can choose not to
    # compile the code using cython, they can instead just use
    # the .c file that's also distributed
    # http://cython.readthedocs.io/en/latest/src/reference/compilation.html#distributing-cython-modules
    src_ext = '.pyx' if use_cython else '.c'
    names = ['pairwise3']
    modules = [Extension(name,
                         [os.path.join(name + src_ext)],
                         extra_compile_args = compile_args,
                         extra_link_args = link_args) for name in names]

    if use_cython:
        return cythonize(modules)
    else:
        return modules


USE_OPENMP = set_gcc(USE_OPENMP)
setup(
    name = NAME,
    version = VERSION,
    description = 'pairwise distance quickstart',
    ext_modules = define_extensions(use_cython, USE_OPENMP),
    include_dirs = [np.get_include()]
)
