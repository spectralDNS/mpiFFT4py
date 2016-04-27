__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2016-04-14"
__copyright__ = "Copyright (C) 2016 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

import numpy as np
from mpi4py import MPI
import collections

class work_array_dict(dict):
    """Dictionary of work arrays indexed by their shape, type and an indicator i.
    """
    def __missing__(self, key):
        shape, dtype, i = key
        a = np.zeros(shape, dtype=dtype)
        self[key] = a
        return self[key]

class work_arrays(collections.MutableMapping):
    """A dictionary to hold numpy work arrays.
    
    The dictionary allows two types of keys for the same item.    
    The key may be a numpy array and an index, like (ndarray, index), 
    where the shape and type is implied, or
    the key may be a tuple of (shape, dtype, index)
    shape is a tuple
    """

    def __init__(self, *args, **kwargs):
        self.store = work_array_dict()
        self.update(dict(*args, **kwargs))  # use the free update to set keys

    def __getitem__(self, key):
        return self.store[self.__keytransform__(key)]

    def __setitem__(self, key, value):
        self.store[self.__keytransform__(key)] = value

    def __delitem__(self, key):
        del self.store[self.__keytransform__(key)]

    def __iter__(self):
        raise TypeError('Work arrays not iterable')

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
    """Return datatypes associated with precision.
    """
    assert precision in ("single", "double")
    return {"single": (np.float32, np.complex64, MPI.F_FLOAT_COMPLEX),
            "double": (np.float64, np.complex128, MPI.F_DOUBLE_COMPLEX)}[precision]

