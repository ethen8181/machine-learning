import os
import re
import sys
import glob
import logging
import numpy as np
from setuptools import Extension, setup, find_packages
try:
    from Cython.Build import cythonize
    USE_CYTHON = True
except ImportError:
    USE_CYTHON = False


# top-level information
NAME = 'mlutils'
SRC_ROOT = 'mlutils'
DESCRIPTION = 'machine learning helper/utility functions'
KEYWORDS = 'Machine Learning, Python3'
USE_OPENMP = True
EXTENSIONS = ['models._fm']


def get_version(src_root):
    """
    Look for version, __version__, under
    the __init__.py file of the package.
    """
    file = os.path.join(src_root, '__init__.py')
    with open(file) as f:
        matched = re.findall("__version__ = '([\d.\w]+)'", f.read())
        version = matched[0]

    return version


def define_extensions(extensions, use_cython, use_openmp, src_root):
    """
    Compile the extensions, the only thing that we need to
    worry about is passing all the modules that requires
    compilation.
    """

    # for compiling with c++11, both link_args and
    # compile_args should have -std=c++11
    link_args = ['-std=c++11']
    compile_args = ['-std=c++11']
    if sys.platform.startswith('win'):
        # compile args from
        # https://msdn.microsoft.com/en-us/library/fwkeyyhe.aspx
        compile_args += ['/O2', '/openmp']
    else:
        compile_args += [
            '-Wno-unused-function',
            '-Wno-maybe-uninitialized',
            '-O3', '-ffast-math']

        use_openmp, gcc = set_gcc(use_openmp)
        if gcc is not None:
            # ensure the path to gcc is linked properly
            # https://github.com/gprMax/gprMax/issues/134#issuecomment-340467832
            rpath = '/usr/local/opt/gcc/lib/gcc/' + gcc[-1] + '/'
            link_args.append('-Wl,-rpath,' + rpath)

        if use_openmp:
            link_args.append('-fopenmp')
            compile_args.append('-fopenmp')

    # recommended approach is that the user can choose not to
    # compile the code using cython, they can instead just use
    # the .cpp file that's also distributed
    # http://cython.readthedocs.io/en/latest/src/reference/compilation.html#distributing-cython-modules
    src_ext = '.pyx' if use_cython else '.cpp'

    # Extension name should be '.' delimited and
    # sources name should be ',' delimited
    modules = []
    for name in extensions:
        source_name = name.split('.')
        source_name[-1] += src_ext
        source = os.path.join(src_root, *source_name)
        module = Extension(
            src_root + '.' + name,
            sources = [source],
            language = 'c++',
            include_dirs = [np.get_include()],
            extra_compile_args = compile_args,
            extra_link_args = link_args)
        modules.append(module)

    if use_cython:
        return cythonize(modules)
    else:
        return modules


def set_gcc(use_openmp):
    """
    Try to find and use GCC on OSX for OpenMP support.

    References
    ----------
    - https://github.com/maciejkula/glove-python/blob/master/setup.py
    - https://groups.google.com/forum/#!topic/cython-users/dGxpilCY0p8
    """

    # For macports and homebrew
    patterns = [
        '/opt/local/bin/g++-mp-[0-9].[0-9]',
        '/opt/local/bin/g++-mp-[0-9]',
        '/usr/local/bin/g++-[0-9].[0-9]',
        '/usr/local/bin/g++-[0-9]']

    gcc = None
    if 'darwin' in sys.platform.lower():
        gcc_binaries = []
        for pattern in patterns:
            gcc_binaries += glob.glob(pattern)

        if gcc_binaries:
            use_openmp = True
            gcc_binaries.sort()
            _, gcc = os.path.split(gcc_binaries[-1])
            os.environ['CC'] = gcc
            os.environ["CXX"] = gcc
        else:
            use_openmp = False
            logging.warning('No GCC available. Install gcc from Homebrew: '
                            'brew install gcc --without-multilib')
    return use_openmp, gcc


with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name = NAME,
    version = get_version(SRC_ROOT),
    description = DESCRIPTION,
    keywords = KEYWORDS,
    license = 'MIT',
    author = 'Ethen Liu',
    author_email = 'ethen8181@gmail.com',
    classifiers = [
        'Natural Language :: English',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'License :: OSI Approved :: MIT License'],
    ext_modules = define_extensions(EXTENSIONS, USE_CYTHON, USE_OPENMP, SRC_ROOT),
    setup_requires = ['cython>=0.25.2'],
    install_requires = required,
    packages = find_packages())
