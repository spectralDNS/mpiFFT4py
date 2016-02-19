import pytest
import string
from numpy.random import random
from numpy import allclose, zeros, zeros_like, pi, array
from mpi4py import MPI

from mpiFFT4py.pencil import FastFourierTransform as pencil_FFT
from mpiFFT4py.slab import FastFourierTransform as slab_FFT
from mpiFFT4py.line import FastFourierTransform as line_FFT
from mpiFFT4py import rfft2, rfftn, irfftn, irfft2

N = 2**4
L = array([2*pi, 2*pi, 2*pi])

@pytest.fixture(params=("pencilys", "pencilyd",
                        "slabs", "slabd"), scope='module')
def FFT(request):
    prec = {"s": "single", "d":"double"}[request.param[-1]]
    if request.param[:3] == "pen":
        return pencil_FFT[string.upper(request.param[-2])](array([N, N, N]), L, MPI, prec)
    else:
        return slab_FFT(array([N, N, N]), L, MPI, prec)

@pytest.fixture(params=("lines", "lined"), scope='module')
def FFT2(request):
    prec = {"s": "single", "d":"double"}[request.param[-1]]
    return line_FFT(array([N, N]), L[:-1], MPI, prec)
    
def test_FFT(FFT):
    if FFT.rank == 0:
        A = random((N, N, N)).astype(FFT.float)
    else:
        A = zeros((N, N, N), dtype=FFT.float)

    FFT.comm.Bcast(A, root=0)
    a = zeros(FFT.real_shape(), dtype=FFT.float)
    c = zeros(FFT.complex_shape(), dtype=FFT.complex)
    a[:] = A[FFT.real_local_slice()]
    c = FFT.fftn(a, c)
    B2 = rfftn(A, axes=(0,1,2))
    a = FFT.ifftn(c, a)
    assert allclose(a, A[FFT.real_local_slice()], 5e-7, 5e-7)

def test_FFT2(FFT2):
    if FFT2.rank == 0:
        A = random((N, N)).astype(FFT2.float)
    else:
        A = zeros((N, N), dtype=FFT2.float)

    FFT2.comm.Bcast(A, root=0)
    a = zeros(FFT2.real_shape(), dtype=FFT2.float)
    c = zeros(FFT2.complex_shape(), dtype=FFT2.complex)
    a[:] = A[FFT2.real_local_slice()]
    c = FFT2.fft2(a, c)
    B2 = rfft2(A, axes=(0,1))
    assert allclose(c, B2[FFT2.complex_local_slice()])
    
    a = FFT2.ifft2(c, a)
    assert allclose(a, A[FFT2.real_local_slice()], 5e-6, 5e-6)

#test_FFT(pencil_FFT["Y"](array([N, N, N], dtype=int), L.astype(float64), MPI, "double", 2))
#test_FFT(pencil_FFT["Y"](array([N, N, N], dtype=int), L, MPI, "single", 2))
#test_FFT(slab_FFT(array([N, N, N]), L, MPI, "single"))
#test_FFT2(line_FFT(array([N, N]), L[:-1], MPI, "double"))
