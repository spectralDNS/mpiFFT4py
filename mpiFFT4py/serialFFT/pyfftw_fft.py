__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2016-02-16"
__copyright__ = "Copyright (C) 2016 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

__all__ = ['dct', 'fft', 'ifft', 'fft2', 'ifft2', 'fftn', 'ifftn',
           'rfft', 'irfft', 'rfft2', 'irfft2', 'rfftn', 'irfftn', 
           'fftfreq', 'rfftfreq', 'empty', 'zeros']

import pyfftw
from numpy import iscomplexobj, zeros_like, float64, zeros as npzeros
from numpy.fft import fftfreq, rfftfreq

def empty(N, dtype=float64, bytes=32):
    return pyfftw.n_byte_align_empty(N, bytes, dtype=dtype)

def zeros(N, dtype=float64, bytes=32):
    return pyfftw.n_byte_align(npzeros(N, dtype=dtype), bytes)

dct_object    = {}
fft_object    = {}
ifft_object   = {}
fft2_object   = {}
ifft2_object  = {}
fftn_object   = {}
ifftn_object  = {}
irfft_object  = {}
irfftn_object = {}
irfft2_object = {}
rfft2_object  = {}
rfft_object   = {}
rfftn_object  = {}
    
def ifft(a, axis=None,threads=1):
    global ifft_object
    if not a.shape in ifft_object:
        b = a.copy()
        ifft_object[(a.shape, a.dtype)] = pyfftw.builders.ifft(b, axis=axis,threads=threads)
        
    in_array = ifft_object[(a.shape, a.dtype)].get_input_array()
    in_array[:] = a
    return ifft_object[(a.shape, a.dtype)]()

def ifft2(a, axes=None,threads=1):
    global ifft2_object
    if not a.shape in ifft2_object:
        b = a.copy()
        ifft2_object[(a.shape, a.dtype)] = pyfftw.builders.ifft2(b, axes=axes,threads=threads)    
        
    in_array = ifft2_object[(a.shape, a.dtype)].get_input_array()
    in_array[:] = a
    return ifft2_object[(a.shape, a.dtype)]()

def ifftn(a, axes=None,threads=1):
    global ifftn_object
    if not a.shape in ifftn_object:
        b = a.copy()
        ifftn_object[(a.shape, a.dtype)] = pyfftw.builders.ifftn(b, axes=axes,threads=threads)    
        
    in_array = ifftn_object[(a.shape, a.dtype)].get_input_array()
    in_array[:] = a
    return ifftn_object[(a.shape, a.dtype)]()

def irfft(a, axis=None,threads=1):
    global irfft_object
    if not a.shape in irfft_object:
        b = a.copy()
        irfft_object[(a.shape, a.dtype)] = pyfftw.builders.irfft(b, axis=axis,threads=threads)
        
    in_array = irfft_object[(a.shape, a.dtype)].get_input_array()
    in_array[:] = a
    return irfft_object[(a.shape, a.dtype)]()

def irfft2(a, axes=None,threads=1):
    global irfft2_object
    if not a.shape in irfft2_object:
        b = a.copy()
        irfft2_object[(a.shape, a.dtype)] = pyfftw.builders.irfft2(b, axes=axes,threads=threads)
        
    in_array = irfft2_object[(a.shape, a.dtype)].get_input_array()
    in_array[:] = a
    return irfft2_object[(a.shape, a.dtype)]()

def irfftn(a, axes=None,threads=1):
    global irfftn_object
    if not a.shape in irfftn_object:
        b = a.copy()
        irfftn_object[(a.shape, a.dtype)] = pyfftw.builders.irfftn(b, axes=axes,threads=threads)
        
    in_array = irfftn_object[(a.shape, a.dtype)].get_input_array()
    in_array[:] = a
    return irfftn_object[(a.shape, a.dtype)]()

def fft(a, axis=None,threads=1):
    global fft_object
    if not a.shape in fft_object:
        b = a.copy()
        fft_object[(a.shape, a.dtype)] = pyfftw.builders.fft(b, axis=axis,threads=threads)
    
    in_array = fft_object[(a.shape, a.dtype)].get_input_array()
    in_array[:] = a
    return fft_object[(a.shape, a.dtype)]()

def fft2(a, axes=None,threads=1):
    global fft2_object
    if not a.shape in fft2_object:
        b = a.copy()
        fft2_object[(a.shape, a.dtype)] = pyfftw.builders.fft2(b, axes=axes,threads=threads)
    
    in_array = fft2_object[(a.shape, a.dtype)].get_input_array()
    in_array[:] = a
    return fft2_object[(a.shape, a.dtype)]()

def fftn(a, axes=None,threads=1):
    global fftn_object
    if not a.shape in fftn_object:
        b = a.copy()
        fftn_object[(a.shape, a.dtype)] = pyfftw.builders.fftn(b, axes=axes,threads=threads)
    
    in_array = fftn_object[(a.shape, a.dtype)].get_input_array()
    in_array[:] = a
    return fftn_object[(a.shape, a.dtype)]()

def rfft(a, axis=None,threads=1):
    global rfft_object
    if not a.shape in rfft_object:
        b = a.copy()
        rfft_object[(a.shape, a.dtype)] = pyfftw.builders.rfft(b, axis=axis,threads=threads)
        
    in_array = rfft_object[(a.shape, a.dtype)].get_input_array()
    in_array[:] = a        
    return rfft_object[(a.shape, a.dtype)]()

def rfft2(a, axes=None,threads=1):
    global rfft2_object
    if not a.shape in rfft2_object:
        b = a.copy()
        rfft2_object[(a.shape, a.dtype)] = pyfftw.builders.rfft2(b, axes=axes,threads=threads)
        
    in_array = rfft2_object[(a.shape, a.dtype)].get_input_array()
    in_array[:] = a        
    return rfft2_object[(a.shape, a.dtype)]()

def rfftn(a, axes=None,threads=1):
    global rfftn_object
    if not a.shape in rfftn_object:
        b = a.copy()
        rfftn_object[(a.shape, a.dtype)] = pyfftw.builders.rfftn(b, axes=axes,threads=threads)
        
    in_array = rfftn_object[(a.shape, a.dtype)].get_input_array()
    in_array[:] = a        
    return rfftn_object[(a.shape, a.dtype)]()

#TODO:Figure out how to use threads here
if hasattr(pyfftw.builders, "dchijt"):
    def dct(a, type=2, axis=0):
        global dct_object
        if not (a.shape, type) in dct_object:
            if iscomplexobj(a):
                b = a.real.copy()
            else:
                b = a.copy()
            dct_object[(a.shape, a.dtype, type)] = (pyfftw.builders.dct(b, axis=axis, type=type), a.copy())
            
        dobj, c = dct_object[(a.shape, a.dtype, type)]
        in_array = dobj.get_input_array()
        if iscomplexobj(a):
            in_array[:] = a.real
            c.real[:] = dobj()
            in_array[:] = a.imag
            c.imag[:] = dobj()            

        else:
            in_array[:] = a
            c[:] = dobj()
        return c
    
else:
    dct1 = pyfftw.interfaces.scipy_fftpack.dct
    def dct(x, type=2, axis=0):
        if iscomplexobj(x):
            c = zeros_like(x)
            c.real[:] = dct1(x.real, type=type, axis=axis)
            c.imag[:] = dct1(x.imag, type=type, axis=axis)
            return c

        else:
            return dct1(x, type=type, axis=axis)
