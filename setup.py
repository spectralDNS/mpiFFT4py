#!/usr/bin/env python

import os, sys, platform
from distutils.core import setup, Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
from numpy import get_include

# Version number
major = 1
minor = 0
maintenance = 0

cwd = os.path.abspath(os.path.dirname(__file__))
cdir = os.path.join(cwd, "mpiFFT4py", "cython")

cmdclass = {}
class build_ext_subclass(build_ext):
    def build_extensions(self):
        extra_compile_args = ['-w', '-Ofast']
        cmd = "echo | %s -E - %s &>/dev/null" % (
            self.compiler.compiler[0], " ".join(extra_compile_args))
        try:
            subprocess.check_call(cmd, shell=True)
        except:
            extra_compile_args = ['-w', '-O3']
        for e in self.extensions:
            e.extra_compile_args = extra_compile_args
        build_ext.build_extensions(self)

ext = cythonize(os.path.join(cdir, "*.pyx"))
[e.include_dirs.extend([get_include()]) for e in ext]
cmdclass = {'build_ext': build_ext_subclass}

setup(name = "mpiFFT4py",
      version = "%d.%d.%d" % (major, minor, maintenance),
      description = "mpiFFT4py -- Parallel 3D FFT in Python using MPI for Python",
      long_description = "",
      author = "Mikael Mortensen",
      author_email = "mikaem@math.uio.no", 
      url = 'https://github.com/spectralDNS/mpiFFT4py',
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
                  "mpiFFT4py.cython"
                  ],
      package_dir = {"mpiFFT4py": "mpiFFT4py"}
      ext_modules = ext,
      cmdclass = cmdclass
    )
