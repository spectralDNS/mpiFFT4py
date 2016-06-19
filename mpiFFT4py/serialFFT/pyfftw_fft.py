__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2016-02-16"
__copyright__ = "Copyright (C) 2016 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

__all__ = ['dct', 'fft', 'ifft', 'fft2', 'ifft2', 'fftn', 'ifftn',
           'rfft', 'irfft', 'rfft2', 'irfft2', 'rfftn', 'irfftn', 
           'fftfreq', 'rfftfreq', 'empty', 'zeros']

import pyfftw
from numpy import iscomplexobj, zeros_like, float64, ascontiguousarray, zeros as npzeros
from numpy.fft import fftfreq, rfftfreq
import copy

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

def ifft(a, b=None, axis=None, overwrite_input=False, threads=1, planner_effort="FFTW_MEASURE"):
    global ifft_object
    if not (a.shape, a.dtype, overwrite_input, axis) in ifft_object:
        ifft_object[(a.shape, a.dtype, overwrite_input, axis)] = pyfftw.builders.ifft(a, axis=axis, overwrite_input=overwrite_input, threads=threads, planner_effort=planner_effort) 
    if not b is None:
        if b.flags['C_CONTIGUOUS'] is True:
            ifft_object[(a.shape, a.dtype, overwrite_input, axis)](a, b)
        else:
            ifft_object[(a.shape, a.dtype, overwrite_input, axis)](a)
            b[:] = ifft_object[(a.shape, a.dtype, overwrite_input, axis)].output_array
        return b
    else:
        ifft_object[(a.shape, a.dtype, overwrite_input, axis)](a)
        return ifft_object[(a.shape, a.dtype, overwrite_input, axis)].output_array

def ifft2(a, b=None, axes=None, overwrite_input=False, threads=1, planner_effort="FFTW_MEASURE"):
    global ifft2_object
    if not (a.shape, a.dtype, overwrite_input, axes) in ifft2_object:
        ifft2_object[(a.shape, a.dtype, overwrite_input, axes)] = pyfftw.builders.ifft2(a, axes=axes, overwrite_input=overwrite_input, threads=threads, planner_effort=planner_effort)
    if not b is None:
        if b.flags['C_CONTIGUOUS'] is True:
            ifft2_object[(a.shape, a.dtype, overwrite_input, axes)](a, b)
        else:
            ifft2_object[(a.shape, a.dtype, overwrite_input, axes)](a)
            b[:] = ifft2_object[(a.shape, a.dtype, overwrite_input, axes)].output_array
        return b
    else:
        ifft2_object[(a.shape, a.dtype, overwrite_input, axes)](a)
        return ifft2_object[(a.shape, a.dtype, overwrite_input, axes)].output_array

def ifftn(a, b=None, axes=None, overwrite_input=False, threads=1, planner_effort="FFTW_MEASURE"):
    global ifftn_object
    if not (a.shape, a.dtype, overwrite_input, axes) in ifftn_object:
        ifftn_object[(a.shape, a.dtype, overwrite_input, axes)] = pyfftw.builders.ifftn(a, axes=axes, overwrite_input=overwrite_input, threads=threads, planner_effort=planner_effort)      
    if not b is None:
        if b.flags['C_CONTIGUOUS'] is True:
            ifftn_object[(a.shape, a.dtype, overwrite_input, axes)](a, b)
        else:
            ifftn_object[(a.shape, a.dtype, overwrite_input, axes)](a)
            b[:] = ifftn_object[(a.shape, a.dtype, overwrite_input, axes)].output_array
        return b
    else:
        ifftn_object[(a.shape, a.dtype, overwrite_input, axes)](a)
        return ifftn_object[(a.shape, a.dtype, overwrite_input, axes)].output_array

def irfft(a, b=None, axis=None, overwrite_input=False, threads=1, planner_effort="FFTW_MEASURE"):
    global irfft_object
    if not (a.shape, a.dtype, axis) in irfft_object:
        irfft_object[(a.shape, a.dtype, axis)] = pyfftw.builders.irfft(a, axis=axis, threads=threads, planner_effort=planner_effort)
    if overwrite_input:
        irfft_object[(a.shape, a.dtype, axis)](a)
    else:
        irfft_object[(a.shape, a.dtype, axis)](a.copy())
    if not b is None:
        b[:] = irfft_object[(a.shape, a.dtype, axis)].output_array
        return b
    else:
        return irfft_object[(a.shape, a.dtype, axis)].output_array

def irfft2(a, b=None, axes=None, overwrite_input=False, threads=1, planner_effort="FFTW_MEASURE"):
    global irfft2_object
    if not (a.shape, a.dtype, axes) in irfft2_object:
        irfft2_object[(a.shape, a.dtype, axes)] = pyfftw.builders.irfft2(a, axes=axes, threads=threads, planner_effort=planner_effort)
    # Copy required for irfft2 because input is destroyed
    if overwrite_input:
        irfft2_object[(a.shape, a.dtype, axes)](a)
    else:
        irfft2_object[(a.shape, a.dtype, axes)](a.copy())
    if not b is None:
        b[:] = irfft2_object[(a.shape, a.dtype, axes)].output_array
        return b
    else:
        return irfft2_object[(a.shape, a.dtype, axes)].output_array

def irfftn(a, b=None, axes=None, overwrite_input=False, threads=1, planner_effort="FFTW_MEASURE"):
    global irfftn_object
    if not (a.shape, a.dtype, axes) in irfftn_object:
        irfftn_object[(a.shape, a.dtype, axes)] = pyfftw.builders.irfftn(a, axes=axes, threads=threads, planner_effort=planner_effort)
    # Copy required because input is always destroyed    
    if overwrite_input:
        irfftn_object[(a.shape, a.dtype, axes)](a)
    else:
        irfftn_object[(a.shape, a.dtype, axes)](a.copy())
    if not b is None:
        b[:] = irfftn_object[(a.shape, a.dtype, axes)].output_array
        return b
    else:
        return irfftn_object[(a.shape, a.dtype, axes)].output_array

def fft(a, b=None, axis=None, overwrite_input=False, threads=1, planner_effort="FFTW_MEASURE"):
    global fft_object
    if not (a.shape, a.dtype, overwrite_input, axis) in fft_object:
        fft_object[(a.shape, a.dtype, overwrite_input, axis)] = pyfftw.builders.fft(a, axis=axis, overwrite_input=overwrite_input, threads=threads, planner_effort=planner_effort)
    if not b is None:
        if b.flags['C_CONTIGUOUS'] is True:
            fft_object[(a.shape, a.dtype, overwrite_input, axis)](a, b)
        else:
            fft_object[(a.shape, a.dtype, overwrite_input, axis)](a)
            b[:] = fft_object[(a.shape, a.dtype, overwrite_input, axis)].output_array
        return b
    else:
        fft_object[(a.shape, a.dtype, overwrite_input, axis)](a)
        return fft_object[(a.shape, a.dtype, overwrite_input, axis)].output_array

def fft2(a, b=None, axes=None, overwrite_input=False, threads=1, planner_effort="FFTW_MEASURE"):
    global fft2_object
    if not (a.shape, a.dtype, overwrite_input, axes) in fft2_object:
        fft2_object[(a.shape, a.dtype, overwrite_input, axes)] = pyfftw.builders.fft2(a, axes=axes, overwrite_input=overwrite_input, threads=threads, planner_effort=planner_effort)
    if not b is None:
        if b.flags['C_CONTIGUOUS'] is True:
            fft2_object[(a.shape, a.dtype, overwrite_input, axes)](a, b)
        else:
            fft2_object[(a.shape, a.dtype, overwrite_input, axes)](a)
            b[:] = fft2_object[(a.shape, a.dtype, overwrite_input, axes)].output_array
        return b
    else:
        fft2_object[(a.shape, a.dtype, overwrite_input, axes)](a)
        return fft2_object[(a.shape, a.dtype, overwrite_input, axes)].output_array

def fftn(a, b=None, axes=None, overwrite_input=False, threads=1, planner_effort="FFTW_MEASURE"):
    global fftn_object
    if not (a.shape, a.dtype, overwrite_input, axes) in fftn_object:
        fftn_object[(a.shape, a.dtype, overwrite_input, axes)] = pyfftw.builders.fftn(a, axes=axes, overwrite_input=overwrite_input, threads=threads, planner_effort=planner_effort)    
    if not b is None:
        if b.flags['C_CONTIGUOUS'] is True:
            fftn_object[(a.shape, a.dtype, overwrite_input, axes)](a, b)
        else:
            fftn_object[(a.shape, a.dtype, overwrite_input, axes)](a)
            b[:] = fftn_object[(a.shape, a.dtype, overwrite_input, axes)].output_array
        return b
    else:
        fftn_object[(a.shape, a.dtype, overwrite_input, axes)](a)
        return fftn_object[(a.shape, a.dtype, overwrite_input, axes)].output_array

def rfft(a, b=None, axis=None, overwrite_input=False, threads=1, planner_effort="FFTW_MEASURE"):
    global rfft_object
    if not (a.shape, a.dtype, overwrite_input, axis) in rfft_object:
        rfft_object[(a.shape, a.dtype, overwrite_input, axis)] = pyfftw.builders.rfft(a, axis=axis, overwrite_input=overwrite_input, threads=threads, planner_effort=planner_effort)
    if not b is None:
        if b.flags['C_CONTIGUOUS'] is True:
            rfft_object[(a.shape, a.dtype, overwrite_input, axis)](a, b)
        else:
            rfft_object[(a.shape, a.dtype, overwrite_input, axis)](a)
            b[:] = rfft_object[(a.shape, a.dtype, overwrite_input, axis)].output_array
        return b
    else:
        rfft_object[(a.shape, a.dtype, overwrite_input, axis)](a)
        return rfft_object[(a.shape, a.dtype, overwrite_input, axis)].output_array

def rfft2(a, b=None, axes=None, overwrite_input=False, threads=1, planner_effort="FFTW_MEASURE"):
    global rfft2_object
    if not (a.shape, a.dtype, overwrite_input, axes) in rfft2_object:
        rfft2_object[(a.shape, a.dtype, overwrite_input, axes)] = pyfftw.builders.rfft2(a, axes=axes, overwrite_input=overwrite_input, threads=threads, planner_effort=planner_effort)  
    if not b is None:
        if b.flags['C_CONTIGUOUS'] is True:
            rfft2_object[(a.shape, a.dtype, overwrite_input, axes)](a, b)
        else:
            rfft2_object[(a.shape, a.dtype, overwrite_input, axes)](a)
            b[:] = rfft2_object[(a.shape, a.dtype, overwrite_input, axes)].output_array
        return b
    else:
        rfft2_object[(a.shape, a.dtype, overwrite_input, axes)](a)
        return rfft2_object[(a.shape, a.dtype, overwrite_input, axes)].output_array

def rfftn(a, b=None, axes=None, overwrite_input=False, threads=1, planner_effort="FFTW_MEASURE"):
    global rfftn_object
    if not (a.shape, a.dtype, overwrite_input, axes) in rfftn_object:
        rfftn_object[(a.shape, a.dtype, overwrite_input, axes)] = pyfftw.builders.rfftn(a, axes=axes, overwrite_input=overwrite_input, threads=threads, planner_effort=planner_effort)
    if not b is None:
        if b.flags['C_CONTIGUOUS'] is True:
            rfftn_object[(a.shape, a.dtype, overwrite_input, axes)](a, b)
        else:
            rfftn_object[(a.shape, a.dtype, overwrite_input, axes)](a)
            b[:] = rfftn_object[(a.shape, a.dtype, overwrite_input, axes)].output_array
        return b
    else:
        rfftn_object[(a.shape, a.dtype, overwrite_input, axes)](a)
        return rfftn_object[(a.shape, a.dtype, overwrite_input, axes)].output_array

if hasattr(pyfftw.builders, "dct"):
    #@profile
    def dct(a, b, type=2, axis=0, overwrite_input=False, threads=1, planner_effort="FFTW_EXHAUSTIVE"):
        global dct_object
        key = (a.shape, a.dtype, overwrite_input, axis, type)
        if not key in dct_object:
            if iscomplexobj(a):
                ac = a.real.copy()
            else:
                ac = a
            dct_object[key] = pyfftw.builders.dct(ac, axis=axis, type=type, 
                                                  overwrite_input=overwrite_input, 
                                                  threads=threads,
                                                  planner_effort=planner_effort)
            
        dobj = dct_object[key]
        c = dobj.get_output_array()
        if iscomplexobj(a):
            dobj(a.real, c)
            b.real[:] = c
            dobj(a.imag, c)
            b.imag[:] = c

        else:
            dobj(a)
            b[:] = c
        return b
    
else:
    dct1 = pyfftw.interfaces.scipy_fftpack.dct
    #@profile
    def dct(a, b, type=2, axis=0, overwrite_input=False, threads=1, planner_effort="FFTW_MEASURE"):
        if iscomplexobj(a):
            b.real[:] = dct1(a.real, type=type, axis=axis)
            b.imag[:] = dct1(a.imag, type=type, axis=axis)
            return b

        else:
            b[:] = dct1(a, type=type, axis=axis)
            return b


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

#def rfft(a, b, axis=0, overwrite_input=False, threads=1, planner_effort="FFTW_MEASURE"):
    #b[:] = nfft.rfft(a, axis=axis, overwrite_input=overwrite_input)
    #return b
        
#def irfft(a, b, axis=0, overwrite_input=False, threads=1, planner_effort="FFTW_MEASURE"):
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

#def rfft2(a, b, axes=(0, 1), overwrite_input=False, threads=1, planner_effort="FFTW_MEASURE"):
    #b[:] = nfft.rfft2(a, axes=axes, overwrite_input=overwrite_input)
    #return b
        
#def irfft2(a, b, axes=(0, 1), overwrite_input=False, threads=1, planner_effort="FFTW_MEASURE"):
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

#def rfftn(a, b, axes=(0, 1, 2), overwrite_input=False, threads=1, planner_effort="FFTW_MEASURE"):
    #b[:] = nfft.rfftn(a, axes=axes, overwrite_input=overwrite_input)
    #return b
        
#def irfftn(a, b, axes=(0, 1, 2), overwrite_input=False, threads=1, planner_effort="FFTW_MEASURE"):
    #b[:] = nfft.irfftn(a, axes=axes, overwrite_input=overwrite_input)
    #return b
 
