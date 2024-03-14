#! /usr/bin/env python
"""Toolbox for Fairness-aware Machine Learning."""

# import codecs

import io
import os

from setuptools import find_packages, setup, Command

VERSION = "0.1.0"

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

DISTNAME = "fairtool"
DESCRIPTION = "Toolbox for Fairness-aware Machine Learning."

# with codecs.open("README.rst", encoding="utf-8-sig") as f:
#     LONG_DESCRIPTION = f.read()

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
here = os.path.abspath(os.path.dirname(__file__))
try:
    with io.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
        LONG_DESCRIPTION = "\n" + f.read()
except FileNotFoundError:
    LONG_DESCRIPTION = DESCRIPTION

AUTHOR = "Zhining Liu"
AUTHOR_EMAIL = "zhining.liu@outlook.com"
MAINTAINER = "Zhining Liu"
MAINTAINER_EMAIL = "zhining.liu@outlook.com"
URL = "https://github.com/ZhiningLiu1998/fairtool"
PROJECT_URLS = {
    "Source": "https://github.com/ZhiningLiu1998/fairtool",
}
LICENSE = "MIT"
CLASSIFIERS = [
    "Intended Audience :: Science/Research",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: C",
    "Programming Language :: Python",
    "Topic :: Software Development",
    "Topic :: Scientific/Engineering",
    "Operating System :: Microsoft :: Windows",
    "Operating System :: POSIX",
    "Operating System :: Unix",
    "Operating System :: MacOS",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3 :: Only",
]
INSTALL_REQUIRES = requirements
EXTRAS_REQUIRE = {
    "dev": [
        "black",
        "flake8",
    ],
    "test": [
        "pytest",
        "pytest-cov",
    ],
    "doc": [
        "sphinx",
        "sphinx-gallery",
        "sphinx_rtd_theme",
        "pydata-sphinx-theme",
        "sphinxcontrib-bibtex",
        "numpydoc",
        "torch",
        "pytest",
    ],
}

setup(
    name=DISTNAME,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    maintainer=MAINTAINER,
    maintainer_email=MAINTAINER_EMAIL,
    description=DESCRIPTION,
    license=LICENSE,
    url=URL,
    version=VERSION,
    project_urls=PROJECT_URLS,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    zip_safe=False,  # the package can run out of an .egg file
    classifiers=CLASSIFIERS,
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
)
