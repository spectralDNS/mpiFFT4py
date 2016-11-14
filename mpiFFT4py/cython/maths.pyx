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

def transpose_Uc(np.ndarray[complex_t, ndim=3] Uc_hatT,
                 np.ndarray[complex_t, ndim=4] U_mpi,
                 int num_processes, int Np0, int Np1, int Nf):    
    cdef unsigned int i, j, k, l, kk
    for i in xrange(num_processes): 
        for j in xrange(Np0):
            for k in xrange(i*Np1, (i+1)*Np1):
                kk = k-i*Np1
                for l in xrange(Nf):
                    Uc_hatT[j, k, l] = U_mpi[i, j, kk, l]
    return Uc_hatT

def transpose_Umpi(np.ndarray[complex_t, ndim=4] U_mpi,
                   np.ndarray[complex_t, ndim=3] Uc_hatT,
                   int num_processes, int Np, int Nf):
    cdef unsigned int i,j,k,l,kk
    for i in xrange(num_processes): 
        for j in xrange(Np):
            for kk in xrange(Np):
                k = kk+i*Np  
                for l in xrange(Nf):
                    U_mpi[i,j,kk,l] = Uc_hatT[j,k,l]
    return U_mpi

    #for i in xrange(num_processes): 
        #for j in xrange(Np):
            #for k in xrange(i*Np, (i+1)*Np):
                #kk = k-i*Np  
                #for l in xrange(Nf):
                    #U_mpi[i,j,kk,l] = Uc_hatT[j,k,l]
    #return U_mpi

#def copy_to_padded(np.ndarray[complex_t, ndim=3] fu, 
                   #np.ndarray[complex_t, ndim=3] fp, 
                   #np.ndarray[int, ndim=1] N, int axis=0):
    #if axis == 0:
        #fp[:N[0]/2] = fu[:N[0]/2]
        #fp[-N[0]/2:] = fu[N[0]/2:]
    #elif axis == 1:
        #fp[:, :N[1]/2] = fu[:, :N[1]/2]
        #fp[:, -N[1]/2:] = fu[:, N[1]/2:]
    #elif axis == 2:
        #fp[:, :, :(N[2]/2+1)] = fu[:]        
    #return fp

#def copy_to_padded_c(np.ndarray[complex_t, ndim=3] fu, 
                     #np.ndarray[complex_t, ndim=3] fp, 
                     #np.ndarray[int, ndim=1] N, int axis=0):
    #if axis == 0:
        #fp[:N[0]] = fu[:N[0]]
    #elif axis == 1:
        #fp[:, :N[1]/2] = fu[:, :N[1]/2]
        #fp[:, -N[1]/2:] = fu[:, N[1]/2:]
    #elif axis == 2:
        #fp[:, :, :(N[2]/2+1)] = fu[:]        
    #return fp

