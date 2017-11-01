import os
import re
from setuptools import setup, find_packages


# top-level information
NAME = 'ml_utils'
SRC_ROOT = 'ml_utils'
DESCRIPTION = 'machine learning helper/utility functions'
KEYWORDS = 'Machine Learning, Python3'


def get_version(src_root):
    file = os.path.join(src_root, '__init__.py')
    with open(file) as f:
        matched = re.findall("__version__ = '([\d.\w]+)'", f.read())
        version = matched[0]
        return version


# read in requirement files
with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name = NAME,
    version = get_version(SRC_ROOT),
    description = DESCRIPTION,
    keywords = KEYWORDS,
    author = 'Ethen Liu',
    author_email = 'ethen8181@gmail.com',
    license = 'MIT',
    classifiers = ['Natural Language :: English',
                   'Programming Language :: Python :: 3.5',
                   'License :: OSI Approved :: MIT License'],
    install_requires = required,
    packages = find_packages()
)
