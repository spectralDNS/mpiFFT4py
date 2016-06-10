#cython: boundscheck=False
#cython: wraparound=False
cimport numpy as np

ctypedef fused complex_t:
    np.complex64_t
    np.complex128_t
    
def dealias_filter(np.ndarray[complex_t, ndim=3] fu,
                   np.ndarray[np.uint8_t, ndim=3] dealias):
    cdef unsigned int i, j, k
    cdef np.uint8_t uu
    for i in xrange(dealias.shape[0]):
        for j in xrange(dealias.shape[1]):
            for k in xrange(dealias.shape[2]):
                uu = dealias[i, j, k]
                fu[i, j, k].real *= uu
                fu[i, j, k].imag *= uu
    return fu