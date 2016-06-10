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

def empty(N, dtype=float64, bytes=64):
    return pyfftw.n_byte_align_empty(N, bytes, dtype=dtype)

def zeros(N, dtype=float64, bytes=64):
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

def ifft(a, b, axis=None, overwrite_input=False, threads=1):
    global ifft_object
    if not (a.shape, a.dtype, overwrite_input, axis) in ifft_object:
        ifft_object[(a.shape, a.dtype, overwrite_input, axis)] = pyfftw.builders.ifft(a, axis=axis, overwrite_input=overwrite_input, threads=threads) 
    ifft_object[(a.shape, a.dtype, overwrite_input, axis)](a, b)
    return b

def ifft2(a, b=None, axes=None, overwrite_input=False, threads=1):
    global ifft2_object
    if not (a.shape, a.dtype, overwrite_input, axes) in ifft2_object:
        ifft2_object[(a.shape, a.dtype, overwrite_input, axes)] = pyfftw.builders.ifft2(a, axes=axes, overwrite_input=overwrite_input, threads=threads)
    ifft2_object[(a.shape, a.dtype, overwrite_input, axes)](a, b)
    return b

def ifftn(a, b=None, axes=None, overwrite_input=False, threads=1):
    global ifftn_object
    if not (a.shape, a.dtype, overwrite_input, axes) in ifftn_object:
        ifftn_object[(a.shape, a.dtype, overwrite_input, axes)] = pyfftw.builders.ifftn(a, axes=axes, overwrite_input=overwrite_input, threads=threads)            
    ifftn_object[(a.shape, a.dtype, overwrite_input, axes)](a, b)
    return b

def irfft(a, b, axis=None, overwrite_input=False, threads=1):
    global irfft_object
    if not (a.shape, a.dtype, axis) in irfft_object:
        irfft_object[(a.shape, a.dtype, axis)] = pyfftw.builders.irfft(a, axis=axis, threads=threads)
    if overwrite_input:
        irfft_object[(a.shape, a.dtype, axis)](a, b)
    else:
        irfft_object[(a.shape, a.dtype, axis)](a.copy(), b)
    return b

def irfft2(a, b, axes=None, overwrite_input=False, threads=1):
    global irfft2_object
    if not (a.shape, a.dtype, axes) in irfft2_object:
        irfft2_object[(a.shape, a.dtype, axes)] = pyfftw.builders.irfft2(a, axes=axes, threads=threads)
    # Copy required for irfft2 because input is destroyed
    if overwrite_input:
        irfft2_object[(a.shape, a.dtype, axes)](a, b)
    else:
        irfft2_object[(a.shape, a.dtype, axes)](a.copy(), b)
    return b

def irfftn(a, b, axes=None, overwrite_input=False, threads=1):
    global irfftn_object
    if not (a.shape, a.dtype, axes) in irfftn_object:
        irfftn_object[(a.shape, a.dtype, axes)] = pyfftw.builders.irfftn(a, axes=axes, threads=threads)
    # Copy required because input is always destroyed
    if overwrite_input:
        irfftn_object[(a.shape, a.dtype, axes)](a, b)
    else:
        irfftn_object[(a.shape, a.dtype, axes)](a.copy(), b)
    return b

def fft(a, b, axis=None, overwrite_input=False, threads=1):
    global fft_object
    if not (a.shape, a.dtype, overwrite_input, axis) in fft_object:
        fft_object[(a.shape, a.dtype, overwrite_input, axis)] = pyfftw.builders.fft(a, axis=axis, overwrite_input=overwrite_input, threads=threads)
    fft_object[(a.shape, a.dtype, overwrite_input, axis)](a, b)
    return b

def fft2(a, b, axes=None, overwrite_input=False, threads=1):
    global fft2_object
    if not (a.shape, a.dtype, overwrite_input, axes) in fft2_object:
        fft2_object[(a.shape, a.dtype, overwrite_input, axes)] = pyfftw.builders.fft2(a, axes=axes, overwrite_input=overwrite_input, threads=threads)
    fft2_object[(a.shape, a.dtype, overwrite_input, axes)](a, b)
    return b

def fftn(a, b, axes=None, threads=1):
    global fftn_object
    if not (a.shape, a.dtype, overwrite_input, axes) in fftn_object:
        fftn_object[(a.shape, a.dtype, overwrite_input, axes)] = pyfftw.builders.fftn(a, axes=axes, overwrite_input=overwrite_input, threads=threads)    
    fftn_object[(a.shape, a.dtype, overwrite_input, axes)](a, b)
    return b

def rfft(a, b, axis=None, overwrite_input=False, threads=1):
    global rfft_object
    if not (a.shape, a.dtype, overwrite_input, axis) in rfft_object:
        rfft_object[(a.shape, a.dtype, overwrite_input, axis)] = pyfftw.builders.rfft(a, axis=axis, overwrite_input=overwrite_input, threads=threads)
    rfft_object[(a.shape, a.dtype, overwrite_input, axis)](a, b)
    return b

def rfft2(a, b, axes=None, overwrite_input=False, threads=1):
    global rfft2_object
    if not (a.shape, a.dtype, overwrite_input, axes) in rfft2_object:
        rfft2_object[(a.shape, a.dtype, overwrite_input, axes)] = pyfftw.builders.rfft2(a, axes=axes, overwrite_input=overwrite_input, threads=threads)  
    rfft2_object[(a.shape, a.dtype, overwrite_input, axes)](a, b)
    return b

def rfftn(a, b, axes=None, overwrite_input=False, threads=1):
    global rfftn_object
    if not (a.shape, a.dtype, overwrite_input, axes) in rfftn_object:
        rfftn_object[(a.shape, a.dtype, overwrite_input, axes)] = pyfftw.builders.rfftn(a, axes=axes, overwrite_input=overwrite_input, threads=threads)
    rfftn_object[(a.shape, a.dtype, overwrite_input, axes)](a, b)
    return b

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


#def fft(a, b=None, axis=0):
    #if b is None:
        #b = nfft.fft(a, axis=axis)
    #else:
        #b[:] = nfft.fft(a, axis=axis)
    #return b
        
#def ifft(a, b=None, axis=0):
    #if b is None:
        #b = nfft.ifft(a, axis=axis)
    #else:
        #b[:] = nfft.ifft(a, axis=axis)
    #return b

#def rfft(a, b, axis=0, overwrite_input=False, threads=1):
    #b[:] = nfft.rfft(a, axis=axis, overwrite_input=overwrite_input)
    #return b
        
#def irfft(a, b, axis=0, overwrite_input=False, threads=1):
    #b[:] = nfft.irfft(a, axis=axis, overwrite_input=overwrite_input)
    #return b

#def fft2(a, b=None, axes=(0, 1)):
    #if b is None:
        #b = nfft.fft2(a, axes=axes)
    #else:
        #b[:] = nfft.fft2(a, axes=axes)
    #return b
        
#def ifft2(a, b=None, axes=(0, 1)):
    #if b is None:
        #b = nfft.ifft2(a, axes=axes)
    #else:
        #b[:] = nfft.ifft2(a, axes=axes)
    #return b

#def rfft2(a, b, axes=(0, 1), overwrite_input=False, threads=1):
    #b[:] = nfft.rfft2(a, axes=axes, overwrite_input=overwrite_input)
    #return b
        
#def irfft2(a, b, axes=(0, 1), overwrite_input=False, threads=1):
    #b[:] = nfft.irfft2(a, axes=axes, overwrite_input=overwrite_input)
    #return b

#def fftn(a, b=None, axes=(0, 1, 2)):
    #if b is None:
        #b = nfft.fftn(a, axes=axes)
    #else:
        #b[:] = nfft.fftn(a, axes=axes)
    #return b
        
#def ifftn(a, b=None, axes=(0, 1, 2)):
    #if b is None:
        #b = nfft.ifftn(a, axes=axes)
    #else:
        #b[:] = nfft.ifftn(a, axes=axes)
    #return b

#def rfftn(a, b, axes=(0, 1, 2), overwrite_input=False, threads=1):
    #b[:] = nfft.rfftn(a, axes=axes, overwrite_input=overwrite_input)
    #return b
        
#def irfftn(a, b, axes=(0, 1, 2), overwrite_input=False, threads=1):
    #b[:] = nfft.irfftn(a, axes=axes, overwrite_input=overwrite_input)
    #return b
 
