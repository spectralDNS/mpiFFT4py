# mpiFFT4py

[![Build Status](https://travis-ci.org/spectralDNS/mpiFFT4py.svg?branch=master)](https://travis-ci.org/spectralDNS/mpiFFT4py)
[![Anaconda-Server Badge](https://anaconda.org/spectraldns/mpifft4py/badges/build.svg)](https://anaconda.org/spectraldns/mpifft4py/builds)

mpiFFT4py performs FFTs in parallel in Python. It is developed to be able to do FFTs in parallel on a three-dimensional computational box (a structured grid), but there are also routines for doing the FFTs on a 2D mesh. The FFTs are computed using serial routines from either *numpy.fft* or *pyfftw*, whereas required MPI communications are performed using *mpi4py*. 

