__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2016-02-16"
__copyright__ = "Copyright (C) 2016 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

__all__ = ['dct', 'fft', 'ifft', 'fft2', 'ifft2', 'fftn', 'ifftn',
           'rfft', 'irfft', 'rfft2', 'irfft2', 'rfftn', 'irfftn', 
           'fftfreq', 'empty', 'zeros']

from numpy import empty, zeros, iscomplexobj
from numpy.fft import fftfreq, fft, ifft, fftn, ifftn, rfft, irfft, rfft2, irfft2, rfftn, irfftn, fft2, ifft2
from scipy.fftpack import dct

dct1 = dct
def dct(x, type=2, axis=0):
    if iscomplexobj(x):
        xreal = dct1(x.real, type=type, axis=axis)
        ximag = dct1(x.imag, type=type, axis=axis)
        return xreal + ximag*1j
    else:
        return dct1(x, type=type, axis=axis)
