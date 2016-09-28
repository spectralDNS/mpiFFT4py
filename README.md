# mpiFFT4py

[![Build Status](https://travis-ci.org/spectralDNS/mpiFFT4py.svg?branch=master)](https://travis-ci.org/spectralDNS/mpiFFT4py)
[![CircleCI](https://circleci.com/gh/spectralDNS/mpiFFT4py/tree/master.svg?style=svg)](https://circleci.com/gh/spectralDNS/mpiFFT4py/tree/master)

Description
----------
mpiFFT4py performs FFTs in parallel in Python. It is developed to be able to do FFTs in parallel on a three-dimensional computational box (a structured grid), but there are also routines for doing the FFTs on a 2D mesh. It implements both the *slab* and the *pencil* decompositions.

Installation
-----------
mpiFFT4py requires *numpy* for basic array oparations, [*pyfftw*](https://github.com/pyfftw/pyFFTW) for efficient FFTs and [*mpi4py*](https://bitbucket.org/mpi4py/mpi4py) for MPI communications. However, if *pyfftw* is not found, then the slower *numpy.fft* is used instead. [*cython*](http://cython.org) is used to optimize a few routines. Install using regular python distutils

    python setup.py install --prefix=...
  
To install in place do

    python setup.py build_ext --inplace
    
To install using Anaconda, you may either compile it yourselves using (from the main directory)

    conda build conf/conda
    conda install mpiFFT4py --use-local
    
or use precompiled binaries in the [*spectralDNS*](https://anaconda.org/spectralDNS/mpifft4py) channel on Anaconda cloud

    conda config --add channels spectralDNS
    conda install mpiFFT4py

Authors
-------
mpiFFT4py is developed by

  * Mikael Mortensen

Licence
-------
mpiFFT4py is licensed under the GNU GPL, version 3 or (at your option) any later version. mpiFFT4py is Copyright (2014-2016) by the authors.

Contact
-------
The latest version of this software can be obtained from

  https://github.com/spectralDNS/mpiFFT4py

Please report bugs and other issues through the issue tracker at:

  https://github.com/spectralDNS/mpiFFT4py/issues
