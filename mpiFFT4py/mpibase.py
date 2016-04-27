__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2016-04-14"
__copyright__ = "Copyright (C) 2016 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

import numpy as np
from mpi4py import MPI
import collections

class work_array_dict(dict):
    """Dictionary of work arrays indexed by their shape, type and an indicator i."""
    def __missing__(self, key):
        shape, dtype, i = key
        a = np.zeros(shape, dtype=dtype)
        self[key] = a
        return self[key]

class work_arrays(collections.MutableMapping):
    """A dictionary to hold numpy work arrays.
    
    The dictionary allows two types of keys for the same item.    
    
    keys:
        - (shape, dtype, index), where shape is tuple, dtype is np.dtype and 
                                 index an integer
        - (ndarray, index),      where ndarray is a numpy array and index is
                                 an integer
                                 
    Usage:
        To create two real work arrays of shape (3,3), do:
        - work = workarrays()
        - a = work[((3,3), np.float, 0)]
        - b = work[(a, 1)]

    """

    def __init__(self):
        self.store = work_array_dict()

    def __getitem__(self, key):
        val = self.store[self.__keytransform__(key)]
        val[:] = 0
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
            assert len(key) == 2
            shape = key[0].shape
            dtype = key[0].dtype
            i = key[1]
            
        elif isinstance(key[0], tuple):
            assert len(key) == 3
            shape, dtype, i = key
            
        else:
            raise TypeError("Wrong type of key for work array")
        
        return (shape, np.dtype(dtype), i)


def datatypes(precision):
    """Return datatypes associated with precision."""
    assert precision in ("single", "double")
    return {"single": (np.float32, np.complex64, MPI.F_FLOAT_COMPLEX),
            "double": (np.float64, np.complex128, MPI.F_DOUBLE_COMPLEX)}[precision]

