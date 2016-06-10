__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2016-02-16"
__copyright__ = "Copyright (C) 2016 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

__all__ = ['dct', 'fft', 'ifft', 'fft2', 'ifft2', 'fftn', 'ifftn',
           'rfft', 'irfft', 'rfft2', 'irfft2', 'rfftn', 'irfftn', 
           'fftfreq', 'rfftfreq', 'empty', 'zeros']

from numpy import empty, zeros, iscomplexobj
from numpy.fft import fftfreq, rfftfreq
import numpy.fft
from scipy.fftpack import dct

dct1 = dct
def dct(x, type=2, axis=0):
    if iscomplexobj(x):
        xreal = dct1(x.real, type=type, axis=axis)
        ximag = dct1(x.imag, type=type, axis=axis)
        return xreal + ximag*1j
    else:
        return dct1(x, type=type, axis=axis)

# Define functions taking both input array and output array

def fft(a, b=None, axis=0, threads=1):
    if b is None:
        b = numpy.fft.fft(a, axis=axis)
    else:
        b[:] = numpy.fft.fft(a, axis=axis)
    return b
        
def ifft(a, b=None, axis=0, threads=1):
    if b is None:
        b = numpy.fft.ifft(a, axis=axis)
    else:
        b[:] = numpy.fft.ifft(a, axis=axis)
    return b

def rfft(a, b, axis=0, threads=1):
    b[:] = numpy.fft.rfft(a, axis=axis)
    return b
        
def irfft(a, b, axis=0, threads=1):
    b[:] = numpy.fft.irfft(a, axis=axis)
    return b

def fft2(a, b=None, axes=(0, 1), threads=1):
    if b is None:
        b = numpy.fft.fft2(a, axes=axes)
    else:
        b[:] = numpy.fft.fft2(a, axes=axes)
    return b
        
def ifft2(a, b=None, axes=(0, 1), threads=1):
    if b is None:
        b = numpy.fft.ifft2(a, axes=axes)
    else:
        b[:] = numpy.fft.ifft2(a, axes=axes)
    return b

def rfft2(a, b, axes=(0, 1), threads=1):
    b[:] = numpy.fft.rfft2(a, axes=axes)
    return b
        
def irfft2(a, b, axes=(0, 1), threads=1):
    b[:] = numpy.fft.irfft2(a, axes=axes)
    return b

def fftn(a, b=None, axes=(0, 1, 2), threads=1):
    if b is None:
        b = numpy.fft.fftn(a, axes=axes)
    else:
        b[:] = numpy.fft.fftn(a, axes=axes)
    return b
        
def ifftn(a, b=None, axes=(0, 1, 2), threads=1):
    if b is None:
        b = numpy.fft.ifftn(a, axes=axes)
    else:
        b[:] = numpy.fft.ifftn(a, axes=axes)
    return b

def rfftn(a, b, axes=(0, 1, 2), threads=1):
    b[:] = numpy.fft.rfftn(a, axes=axes)
    return b
        
def irfftn(a, b, axes=(0, 1, 2), threads=1):
    b[:] = numpy.fft.irfftn(a, axes=axes)
    return b
