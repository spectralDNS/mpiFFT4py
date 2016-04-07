import pytest
import string
from numpy.random import random
from numpy import allclose, zeros, zeros_like, pi, array, int, all
from numpy.fft import fftfreq
from mpi4py import MPI

from mpiFFT4py.pencil import FastFourierTransform as pencil_FFT
from mpiFFT4py.slab import FastFourierTransform as slab_FFT
from mpiFFT4py.line import FastFourierTransform as line_FFT
from mpiFFT4py import rfft2, rfftn, irfftn, irfft2

N = 2**5
L = array([2*pi, 2*pi, 2*pi])
ks = (fftfreq(N)*N).astype(int)

@pytest.fixture(params=("pencilys", "pencilyd",
                        "slabs", "slabd"), scope='module')
def FFT(request):
    prec = {"s": "single", "d":"double"}[request.param[-1]]
    if request.param[:3] == "pen":
        return pencil_FFT(array([N, N, N]), L, MPI, prec, None, alignment=string.upper(request.param[-2]))
    else:
        return slab_FFT(array([N, N, N]), L, MPI, prec)

@pytest.fixture(params=("slabs", "slabd"), scope='module')
def FFT_padded(request):
    prec = {"s": "single", "d":"double"}[request.param[-1]]
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
    #B2 = rfftn(A, axes=(0,1,2))
    #assert allclose(c, B2[FFT.complex_local_slice()])
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
    assert allclose(a, A[FFT2.real_local_slice()], 5e-7, 5e-7)

def test_FFT_padded(FFT_padded):
    FFT = FFT_padded
    N = FFT.N
    if FFT.rank == 0:
        A = random(N).astype(FFT.float)
        C = zeros((FFT.global_complex_shape()), dtype=FFT.complex)
        C[:] = rfftn(A, axes=(0,1,2))
        Cp = zeros((3*N[0]/2, 3*N[1]/2, 3*N[2]/4+1), dtype=FFT.complex)
        Nf = N[2]/2+1
        ks = (fftfreq(N[1])*N[1]).astype(int)
        Cp[:N[0]/2, ks, :Nf] = C[:N[0]/2]
        Cp[-N[0]/2:, ks, :Nf] = C[N[0]/2:]
        Cp[:, -N[1]/2, 0] *= 2
        Cp[-N[0]/2, ks, 0] *= 2
        Cp[-N[0]/2, -N[1]/2, 0] /= 2
        Ap = zeros((3*N[0]/2, 3*N[1]/2, 3*N[2]/2), dtype=FFT.float)
        Ap[:] = irfftn(Cp*1.5**3, axes=(0,1,2))
        
    else:
        C = zeros(FFT.global_complex_shape(), dtype=FFT.complex)
        Ap = zeros((3*N[0]/2, 3*N[1]/2, 3*N[2]/2), dtype=FFT.float)
        
    FFT.comm.Bcast(C, root=0)
    FFT.comm.Bcast(Ap, root=0)
    
    ae = zeros(FFT.real_shape_padded(), dtype=FFT.float)
    c = zeros(FFT.complex_shape(), dtype=FFT.complex)
    
    c[:] = C[FFT.complex_local_slice()]
    ae[:] = Ap[FFT.real_local_slice(padded=True)]
    
    ap = zeros(FFT.real_shape_padded(), dtype=FFT.float)
    cp = zeros(FFT.complex_shape(), dtype=FFT.complex)
    ap = FFT.ifftn(c, ap, padded=True)
    
    assert allclose(ap, ae, 5e-7, 5e-7)
    
    cp = FFT.fftn(ap, cp, padded=True)    
    
    assert all((cp-c)/cp < 5e-5)


#test_FFT(pencil_FFT(array([N, N, N], dtype=int), L, MPI, "double", 2), alignment="X")
#test_FFT(slab_FFT(array([N, N, N]), L, MPI, "single"))
#test_FFT2(line_FFT(array([N, N]), L[:-1], MPI, "single"))
#test_FFT_padded(slab_FFT(array([N, N, N]), L, MPI, "double"))
