__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2016-03-09"
__copyright__ = "Copyright (C) 2016 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from numpy import *
from mpi4py import MPI
#from mpiFFT4py.pencil import R2C
from mpiFFT4py.slab import R2C
from mpi4py_fft.mpifft import PFFT
from time import time

#assert MPI.COMM_WORLD.Get_size() >= 4

# Set global size of the computational box
M = 6
N = array([2**M, 2**M, 2**M], dtype=int)
L = array([2*pi, 2*pi, 2*pi], dtype=float)

# Create an instance of the R2C class for performing 3D FFTs in parallel
# on a cube of size N points and physical size L. The mesh decomposition is
# performed by the FFT class using a slab decomposition. With slab decomposition
# the first index in real physical space is shared amongst the processors,
# whereas in wavenumber space the second index is shared.

#FFT = R2C(N, L, MPI.COMM_WORLD, "double", None, alignment='X', communication='Alltoallw')
FFT = R2C(N, L, MPI.COMM_WORLD, "double", communication='Alltoallw')
fft = PFFT(MPI.COMM_WORLD, N)

U = random.random(FFT.real_shape()).astype(FFT.float) # real_shape = (N[0]//comm.Get_size(), N[1], N[2])
U_copy = zeros_like(U)
U_hat = zeros(FFT.complex_shape(), dtype=FFT.complex) # complex_shape = (N[0], N[1]//comm.Get_size(), N[2]//2+1)

# Perform forward FFT. Real transform in third direction, complex in first two
U_hat = FFT.fftn(U, U_hat)

# Perform inverse FFT.
U_copy = FFT.ifftn(U_hat, U_copy)
MPI.COMM_WORLD.barrier()
t0 = time()
U_hat = FFT.fftn(U, U_hat)
U_copy = FFT.ifftn(U_hat, U_copy)
print "mpiFFT4py ", time()-t0
###########
u = random.random(fft.forward.input_array.shape).astype(fft.forward.input_array.dtype)
MPI.COMM_WORLD.barrier()
t0 = time()
u_hat = fft.forward(u)
u_copy = fft.backward(u_hat)
print "mpi4py-fft ", time()-t0
#########

tol = 1e-6 if FFT.float == float32 else 1e-10

assert allclose(U, U_copy, tol, tol)
assert allclose(u, u_copy, tol, tol)
