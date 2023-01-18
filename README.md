# RRCGAN 
Submitted to Nature Computational Science.

[![Coverage Status](https://coveralls.io/repos/github/neurodata/mgcpy/badge.svg?branch=master)](https://coveralls.io/github/neurodata/mgcpy?branch=master)
[![Build Status](https://travis-ci.com/neurodata/mgcpy.svg?branch=master)](https://travis-ci.com/neurodata/mgcpy)
[![PyPI](https://img.shields.io/pypi/v/mgcpy.svg)](https://pypi.org/project/mgcpy/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/mgcpy.svg)](https://pypi.org/project/mgcpy/)
[![DOI](https://zenodo.org/badge/147731955.svg)](https://zenodo.org/badge/latestdoi/147731955)
[![Documentation Status](https://readthedocs.org/projects/mgcpy/badge/?version=latest)](https://mgcpy.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![PEP8](https://img.shields.io/badge/code%20style-pep8-orange.svg)](https://www.python.org/dev/peps/pep-0008/)
[![Code Climate](https://api.codeclimate.com/v1/badges/51ac28d51f1474bf3567/maintainability)](https://codeclimate.com/github/neurodata/mgcpy/maintainability)

`mgcpy` is a Python package containing tools for independence testing using multiscale graph correlation and other statistical tests, that is capable of dealing with high dimensional and multivariate data.

- [Overview](#overview)
- [Documentation](#documentation)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [Setting up the development environment](#setting-up-the-development-environment)
- [License](#license)
- [Issues](https://github.com/neurodata/mgcpy/issues)

# Overview
``mgcpy`` aims to be a comprehensive independence testing package including commonly used independence tests and additional functionality such as two sample independence testing and a novel random forest-based independence test. These tests are not only included to benchmark MGC but to have a convenient location for users if they would prefer to utilize those tests instead. The package utilizes a simple class structure to enhance usability while also allowing easy extension of the package for developers. The package can be installed on all major platforms (e.g. BSD, GNU/Linux, OS X, Windows)from Python Package Index (PyPI) and GitHub.

