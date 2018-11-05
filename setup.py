#!/usr/bin/env python

import os
import re
import subprocess
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from numpy import get_include

cwd = os.path.abspath(os.path.dirname(__file__))
cdir = os.path.join(cwd, "mpiFFT4py", "cython")

def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    devnull = open(os.devnull, "w")
    p = subprocess.Popen([compiler.compiler[0], '-E', '-'] + [flagname],
                         stdin=subprocess.PIPE, stdout=devnull, stderr=devnull,
                         shell=True)
    p.communicate("")
    return True if p.returncode == 0 else False

class build_ext_subclass(build_ext):
    def build_extensions(self):
        extra_compile_args = ['-g0']
        for c in ['-w', '-Ofast', '-ffast-math', '-march=native']:
            if has_flag(self.compiler, c):
                extra_compile_args.append(c)

        for e in self.extensions:
            e.extra_compile_args += extra_compile_args
            e.include_dirs.extend([get_include()])
        build_ext.build_extensions(self)

ext = [Extension('mpiFFT4py.cython.maths',
                 sources=[os.path.join(cdir, "maths.pyx")])]

def version():
    srcdir = os.path.join(cwd, 'mpiFFT4py')
    with open(os.path.join(srcdir, '__init__.py')) as f:
        m = re.search(r"__version__\s*=\s*'(.*)'", f.read())
        return m.groups()[0]

with open("README.rst", "r") as fh:
    long_description = fh.read()

setup(name = "mpiFFT4py",
      version = version(),
      description = "mpiFFT4py -- Parallel 3D FFT in Python using MPI for Python",
      long_description = long_description,
      author = "Mikael Mortensen",
      author_email = "mikaem@math.uio.no",
      url = 'https://github.com/spectralDNS/mpiFFT4py',
      classifiers = [
          'Development Status :: 5 - Production/Stable',
          'Environment :: Console',
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'Intended Audience :: Education',
          'Programming Language :: Python',
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 3',
          'License :: OSI Approved :: GNU Library or Lesser General Public License (LGPL)',
          'Topic :: Scientific/Engineering :: Mathematics',
          'Topic :: Software Development :: Libraries :: Python Modules',
          ],
      packages = ["mpiFFT4py",
                  "mpiFFT4py.serialFFT",
                  "mpiFFT4py.cython"
                  ],
      package_dir = {"mpiFFT4py": "mpiFFT4py"},
      install_requires=["numpy"],
      setup_requires=["numpy>=1.11",
                      "cython>=0.25",
                      "setuptools>=18.0"],
      ext_modules = ext,
      cmdclass = {'build_ext': build_ext_subclass}
    )
