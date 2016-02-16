#!/usr/bin/env python

import os, sys, platform
from distutils.core import setup, Extension
    
# Version number
major = 1
minor = 0
maintenance = 0

setup(name = "mpiFFT4py",
      version = "%d.%d.%d" % (major, minor, maintenance),
      description = "mpiFFT4py -- Parallel 3D FFT in Python using mpi4py",
      long_description = "",
      author = "Mikael Mortensen",
      author_email = "mikaem@math.uio.no", 
      url = 'https://github.com/mikaem/mpiFFT4py',
      classifiers = [
          'Development Status :: 5 - Production/Stable',
          'Environment :: Console',
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'Intended Audience :: Education',
          'Programming Language :: Python :: 2.7',
          'License :: OSI Approved :: GNU Library or Lesser General Public License (LGPL)',
          'Topic :: Scientific/Engineering :: Mathematics',
          'Topic :: Software Development :: Libraries :: Python Modules',
          ],
      packages = ["mpiFFT4py",
                  "mpiFFT4py.serialFFT",
                  ],
      package_dir = {"mpiFFT4py": "mpiFFT4py"}
    )
