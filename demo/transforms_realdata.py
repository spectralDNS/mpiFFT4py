__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2016-03-09"
__copyright__ = "Copyright (C) 2016 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from numpy import *
from mpi4py import MPI
#from mpiFFT4py.pencil import R2C
from mpiFFT4py.slab import R2C

#assert MPI.COMM_WORLD.Get_size() >= 4

# Set global size of the computational box
M = 6
N = array([2**M, 2**M, 2**M], dtype=int)
L = array([2*pi, 2*pi, 2*pi], dtype=float)

# Create an instance of the R2C class for performing 3D FFTs in parallel
# on a cube of size N points and physical size L. The mesh decomposition is performed by 
# the FFT class using a slab decomposition. With slab decomposition the first index in real
# physical space is shared amongst the processors, whereas in wavenumber space the second 
# index is shared.
#FFT = R2C(N, L, MPI, "double", None, alignment='Y')
FFT = R2C(N, L, MPI, "double", communication='alltoall')

U = random.random(FFT.real_shape()).astype(FFT.float) # real_shape = (N[0]/comm.Get_size(), N[1], N[2])
U_hat = zeros(FFT.complex_shape(), dtype=FFT.complex) # complex_shape = (N[0], N[1]//comm.Get_size(), N[2]/2+1)

# Perform forward FFT. Real transform in third direction, complex in remaining two
for i in range(10):
    U_hat = FFT.fftn(U, U_hat)

# Perform inverse FFT. 
U_copy = zeros_like(U)
U_copy = FFT.ifftn(U_hat, U_copy)

assert allclose(U, U_copy)
