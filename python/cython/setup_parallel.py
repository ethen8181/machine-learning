# usually the name should only be setup.py
# on the terminal run
# python setup_parallel.py install
import os
import sys
import glob
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
    try to find and use GCC on OSX for OpenMP support.
    
    Reference
    ---------
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
            logging.warning('No GCC available. Install gcc from Homebrew '
                            'using brew install gcc.')
    return use_openmp

def define_extensions(use_cython, use_openmp):
    if sys.platform.startswith('win'):
        # compile args from
        # https://msdn.microsoft.com/en-us/library/fwkeyyhe.aspx
        link_args = []
        compile_args = ['/O2', '/openmp']
    else:
        link_args = []
        compile_args = ['-Wno-unused-function', '-Wno-maybe-uninitialized','-O3', '-ffast-math']    
        if use_openmp:
            compile_args.append('-fopenmp')
            link_args.append('-fopenmp')

        if 'anaconda' not in sys.version.lower():
            compile_args.append('-march=native')

    src_ext = '.pyx' if use_cython else '.c'
    modules = [ Extension('pairwise3',
                          [os.path.join('pairwise3' + src_ext)],
                          extra_compile_args = compile_args, 
                          extra_link_args = link_args)]

    if use_cython:
        return cythonize(modules)
    else:
        return modules


USE_OPENMP = set_gcc(USE_OPENMP)
setup(
    name = NAME,
    version = VERSION,
    description = 'pairwise distance quickstart',
    ext_modules = define_extensions(use_cython, USE_OPENMP)
)

