__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2016-02-16"
__copyright__ = "Copyright (C) 2016 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from_numpy = ['fft', 'ifft', 'fft2', 'ifft2', 'fftn', 'ifftn',
           'rfft', 'irfft', 'rfft2', 'irfft2', 'rfftn', 'irfftn', 
           'fftfreq', 'rfftfreq']

__all__ = ['dct', 'fft', 'ifft', 'fft2', 'ifft2', 'fftn', 'ifftn',
           'rfft', 'irfft', 'rfft2', 'irfft2', 'rfftn', 'irfftn', 
           'fftfreq', 'rfftfreq', 'empty', 'zeros']

from numpy import empty, zeros, iscomplexobj
import numpy.fft
from scipy.fftpack import dct

def get_function_from_numpy(what):
    torun = getattr(numpy.fft,what)
    def what_with_threads(*args,**kwargs):
        if "threads" in kwargs.keys():
            del kwargs["threads"]
        return torun(*args,**kwargs)
    globals().update({what:what_with_threads})

map(get_function_from_numpy,from_numpy)

dct1 = dct
def dct(x, type=2, axis=0):
    if iscomplexobj(x):
        xreal = dct1(x.real, type=type, axis=axis)
        ximag = dct1(x.imag, type=type, axis=axis)
        return xreal + ximag*1j
    else:
        return dct1(x, type=type, axis=axis)
