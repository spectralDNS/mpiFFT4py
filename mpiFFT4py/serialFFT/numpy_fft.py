__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2016-02-16"
__copyright__ = "Copyright (C) 2016 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

__all__ = ['dct', 'fft', 'ifft', 'fft2', 'ifft2', 'fftn', 'ifftn',
           'rfft', 'irfft', 'rfft2', 'irfft2', 'rfftn', 'irfftn']

from numpy import iscomplexobj
import numpy.fft
from scipy.fftpack import dct

dct1 = dct
def dct(a, b, type=2, axis=0, **kw):
    if iscomplexobj(a):
        b.real[:] = dct1(a.real, type=type, axis=axis)
        b.imag[:] = dct1(a.imag, type=type, axis=axis)
        return b

    else:
        b[:] = dct1(a, type=type, axis=axis)
        return b

# Define functions taking both input array and output array

def fft(a, b=None, axis=0, threads=1, **kw):
    if b is None:
        return numpy.fft.fft(a, axis=axis)
    else:
        b[:] = numpy.fft.fft(a, axis=axis)
        return b
        
def ifft(a, b=None, axis=0, threads=1, **kw):
    if b is None:
        return numpy.fft.ifft(a, axis=axis)
    else:
        b[:] = numpy.fft.ifft(a, axis=axis)
        return b

def rfft(a, b=None, axis=0, threads=1, **kw):
    if b is None:
        return numpy.fft.rfft(a, axis=axis)
    else:
        b[:] = numpy.fft.rfft(a, axis=axis)
        return b
        
def irfft(a, b=None, axis=0, threads=1, **kw):
    if b is None:
        return numpy.fft.irfft(a, axis=axis)
    else:
        b[:] = numpy.fft.irfft(a, axis=axis)
        return b

def fft2(a, b=None, axes=(0, 1), threads=1, **kw):
    if b is None:
        return numpy.fft.fft2(a, axes=axes)
    else:
        b[:] = numpy.fft.fft2(a, axes=axes)
        return b
        
def ifft2(a, b=None, axes=(0, 1), threads=1, **kw):
    if b is None:
        return numpy.fft.ifft2(a, axes=axes)
    else:
        b[:] = numpy.fft.ifft2(a, axes=axes)
        return b

def rfft2(a, b=None, axes=(0, 1), threads=1, **kw):
    if b is None:
        return numpy.fft.rfft2(a, axes=axes)
    else:
        b[:] = numpy.fft.rfft2(a, axes=axes)
        return b
        
def irfft2(a, b=None, axes=(0, 1), threads=1, **kw):
    if b is None:
        return numpy.fft.irfft2(a, axes=axes)
    else:
        b[:] = numpy.fft.irfft2(a, axes=axes)
        return b

def fftn(a, b=None, axes=(0, 1, 2), threads=1, **kw):
    if b is None:
        return numpy.fft.fftn(a, axes=axes)
    else:
        b[:] = numpy.fft.fftn(a, axes=axes)
        return b
        
def ifftn(a, b=None, axes=(0, 1, 2), threads=1, **kw):
    if b is None:
        return numpy.fft.ifftn(a, axes=axes)
    else:
        b[:] = numpy.fft.ifftn(a, axes=axes)
        return b

def rfftn(a, b=None, axes=(0, 1, 2), threads=1, **kw):
    if b is None:
        return numpy.fft.rfftn(a, axes=axes)
    else:
        b[:] = numpy.fft.rfftn(a, axes=axes)
        return b
        
def irfftn(a, b=None, axes=(0, 1, 2), threads=1, **kw):
    if b is None:
        return numpy.fft.irfftn(a, axes=axes)
    else:
        b[:] = numpy.fft.irfftn(a, axes=axes)
        return b
