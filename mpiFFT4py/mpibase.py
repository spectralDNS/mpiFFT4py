__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2016-04-14"
__copyright__ = "Copyright (C) 2016 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

import numpy as np
from mpi4py import MPI

class _work_arrays(dict):
    """Dictionary of work arrays indexed by their shape, type and an indicator i.
    """
    def __missing__(self, key):
        shape, dtype, i = key
        a = np.zeros(shape, dtype=dtype)
        self[key] = a
        return self[key]

work_arrays = _work_arrays()

def datatypes(precision):
    """Return datatypes associated with precision.
    """
    assert precision in ("single", "double")
    return {"single": (np.float32, np.complex64, MPI.F_FLOAT_COMPLEX),
            "double": (np.float64, np.complex128, MPI.F_DOUBLE_COMPLEX)}[precision]

