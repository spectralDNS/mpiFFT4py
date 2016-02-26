__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2016-02-26"
__copyright__ = "Copyright (C) 2016 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from numpy import *
from mpi4py import MPI
from mpiFFT4py.slab import FastFourierTransform

# Set viscosity, end time and time step
nu = 0.000625
T = 0.1
dt = 0.01

# Set global size of the computational box
N = array([2**5, 2**5, 2**5], dtype=int)
L = array([2*pi, 2*pi, 2*pi], dtype=float)

# Create an instance of the FastFourierTransform clas for performing 3D FFTs in parallel
# on a cube of size N points and physical size L. The mesh decomposition is performed by 
# the FFT class using a slab decomposition. With slab decomposition the first index in real
# physical space is shared amongst the processors, whereas in wavenumber space the second 
# index is shared.
FFT = FastFourierTransform(N, L, MPI, "double")

U     = empty((3,) + FFT.real_shape())                    # real_shape = (N[0]/comm.Get_size(), N[1], N[2])
U_hat = empty((3,) + FFT.complex_shape(), dtype=complex)  # complex_shape = (N[0], N[1]//comm.Get_size(), N[2]/2+1)
P     = empty(FFT.real_shape())
P_hat = empty(FFT.complex_shape(), dtype=complex)
U_hat0  = empty((3,) + FFT.complex_shape(), dtype=complex)
U_hat1  = empty((3,) + FFT.complex_shape(), dtype=complex)
dU      = empty((3,) + FFT.complex_shape(), dtype=complex)
curl    = empty((3,) + FFT.real_shape())
X = FFT.get_local_mesh()
K = FFT.get_scaled_local_wavenumbermesh()
K2 = sum(K*K, 0, dtype=float)
K_over_K2 = K.astype(float) / where(K2==0, 1, K2).astype(float)    
dealias = FFT.get_dealias_filter()
a = [1./6., 1./3., 1./3., 1./6.]
b = [0.5, 0.5, 1.]

def Cross(a, b, c):
    c[0] = FFT.fftn(a[1]*b[2]-a[2]*b[1], c[0])
    c[1] = FFT.fftn(a[2]*b[0]-a[0]*b[2], c[1])
    c[2] = FFT.fftn(a[0]*b[1]-a[1]*b[0], c[2])
    return c

def Curl(a, c):
    c[2] = FFT.ifftn(1j*(K[0]*a[1]-K[1]*a[0]), c[2])
    c[1] = FFT.ifftn(1j*(K[2]*a[0]-K[0]*a[2]), c[1])
    c[0] = FFT.ifftn(1j*(K[1]*a[2]-K[2]*a[1]), c[0])
    return c

def computeRHS(dU, rk):
    if rk > 0:
        for i in range(3):
            U[i] = FFT.ifftn(U_hat[i], U[i])
    curl[:] = Curl(U_hat, curl)
    dU = Cross(U, curl, dU)
    dU *= dealias
    P_hat[:] = sum(dU*K_over_K2, 0, out=P_hat)
    dU -= P_hat*K
    dU -= nu*K2*U_hat
    return dU

# Initialize a Taylor Green vortex
U[0] = sin(X[0])*cos(X[1])*cos(X[2])
U[1] =-cos(X[0])*sin(X[1])*cos(X[2])
U[2] = 0
for i in range(3):
    U_hat[i] = FFT.fftn(U[i], U_hat[i])

# Integrate using a 4th order Rung-Kutta method
t = 0.0
tstep = 0
while t < T-1e-8:
    t += dt; tstep += 1
    U_hat1[:] = U_hat0[:] = U_hat
    for rk in range(4):
        dU = computeRHS(dU, rk)
        if rk < 3: 
            U_hat[:] = U_hat0 + b[rk]*dt*dU
        U_hat1[:] += a[rk]*dt*dU
    U_hat[:] = U_hat1[:]
    for i in range(3):
        U[i] = FFT.ifftn(U_hat[i], U[i])

k = FFT.comm.reduce(sum(U*U)/N[0]/N[1]/N[2]/2)
if FFT.rank == 0:
    assert round(k - 0.124953117517, 7) == 0
