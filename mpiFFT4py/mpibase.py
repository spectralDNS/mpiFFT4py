__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2016-04-14"
__copyright__ = "Copyright (C) 2016 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

import numpy as np
from mpi4py import MPI
import collections

# Possible way to give numpy arrays attributes...
#class Empty(np.ndarray):
    #"""Numpy empty array with additional info dictionary to hold attributes
    #"""
    #def __new__(subtype, shape, dtype=np.float, info={}):
        #obj = np.ndarray.__new__(subtype, shape, dtype)
        #obj.info = info
        #return obj

    #def __array_finalize__(self, obj):
        #if obj is None: return
        #self.info = getattr(obj, 'info', {})

#class Zeros(np.ndarray):
    #"""Numpy zeros array with additional info dictionary to hold attributes
    #"""
    #def __new__(subtype, shape, dtype=float, info={}):
        #obj = np.ndarray.__new__(subtype, shape, dtype)
        #obj.fill(0)
        #obj.info = info
        #return obj

    #def __array_finalize__(self, obj):
        #if obj is None: return
        #self.info = getattr(obj, 'info', {})

Empty, Zeros = np.empty, np.zeros

try:
    import pyfftw
    def empty(N, dtype=np.float, bytes=16):
        return pyfftw.byte_align(Empty(N, dtype=dtype), n=bytes)

    def zeros(N, dtype=np.float, bytes=16):
        return pyfftw.byte_align(Zeros(N, dtype=dtype), n=bytes)
        
except ImportError:
    def empty(N, dtype=np.float, bytes=None):
        return Empty(N, dtype=dtype)

    def zeros(N, dtype=np.float, bytes=None):
        return Zeros(N, dtype=dtype)

class work_array_dict(dict):
    """Dictionary of work arrays indexed by their shape, type and an indicator i."""
    def __missing__(self, key):
        shape, dtype, i = key
        a = zeros(shape, dtype=dtype)
        self[key] = a
        return self[key]

class work_arrays(collections.MutableMapping):
    """A dictionary to hold numpy work arrays.
    
    The dictionary allows two types of keys for the same item.    
    
    keys:
        - (shape, dtype, index (, fillzero)), where shape is tuple, dtype is np.dtype and 
                                              index an integer
        - (ndarray, index (, fillzero)),      where ndarray is a numpy array and index is
                                              an integer
                                              fillzero is an optional bool that determines
                                              whether the array is initialised to zero
                                              
    Usage:
        To create two real work arrays of shape (3,3), do:
        - work = workarrays()
        - a = work[((3,3), np.float, 0)]
        - b = work[(a, 1)]

    Returns:
        Numpy array of given shape. The array is by default initialised to zero, but this
        can be overridden using the fillzero argument.
        
    """

    def __init__(self):
        self.store = work_array_dict()
        self.fillzero = True
    
    def __getitem__(self, key):
        val = self.store[self.__keytransform__(key)]
        if self.fillzero is True: val.fill(0)
        return val

    def __setitem__(self, key, value):
        self.store[self.__keytransform__(key)] = value

    def __delitem__(self, key):
        del self.store[self.__keytransform__(key)]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)
    
    def values(self):
        raise TypeError('Work arrays not iterable')

    def __keytransform__(self, key):
        if isinstance(key[0], np.ndarray):
            if len(key) == 2:
                shape = key[0].shape
                dtype = key[0].dtype
                i = key[1]
                zero = True
                
            elif len(key) == 3:
                shape = key[0].shape
                dtype = key[0].dtype
                i = key[1]
                zero = key[2]
            
        elif isinstance(key[0], tuple):
            if len(key) == 3:
                shape, dtype, i = key
                zero = True
                
            elif len(key) == 4:
                shape, dtype, i, zero = key
            
        else:
            raise TypeError("Wrong type of key for work array")
        
        assert isinstance(zero, bool)
        assert isinstance(i, int)
        self.fillzero = zero
        return (shape, np.dtype(dtype), i)

def datatypes(precision):
    """Return datatypes associated with precision."""
    assert precision in ("single", "double")
    return {"single": (np.float32, np.complex64, MPI.C_FLOAT_COMPLEX),
            "double": (np.float64, np.complex128, MPI.C_DOUBLE_COMPLEX)}[precision]
