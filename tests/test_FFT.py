import pytest
import string
import numpy as np
from numpy.random import random, randn
from numpy import allclose, empty, zeros, zeros_like, pi, array, int, all, float64
from numpy.fft import fftfreq
from mpi4py import MPI

from mpiFFT4py.pencil import R2C as Pencil_R2C
from mpiFFT4py.slab import R2C as Slab_R2C
from mpiFFT4py.line import R2C as Line_R2C
from mpiFFT4py import rfft2, rfftn, irfftn, irfft2, fftn, ifftn, irfft, ifft
from mpiFFT4py.slab import C2C

N = 2**6
L = array([2*pi, 2*pi, 2*pi])
ks = (fftfreq(N)*N).astype(int)

@pytest.fixture(params=("pencilsys", "pencilsyd", "pencilnys", "pencilnyd", 
                        "pencilsxd", "pencilsxs", "pencilnxd", "pencilnxs", 
                        "pencilaxd", "pencilaxs", "pencilayd", "pencilays",
                        "slabas", "slabad", "slabws", "slabwd"), scope='module')
def FFT(request):
    prec = {"s": "single", "d":"double"}[request.param[-1]]
    if request.param[:3] == "pen":
        communication = {"s": "Swap", "n": "Nyquist", "a": "Alltoallw"}[request.param[-3]]
        alignment = string.upper(request.param[-2])
        return Pencil_R2C(array([N, N, N]), L, MPI, prec, communication=communication, alignment=alignment)
    else:
        comm = 'alltoall' if request.param[-2] == 'a' else 'Alltoallw'
        return Slab_R2C(array([N, N, N]), L, MPI, prec, communication=comm)
        
@pytest.fixture(params=("lines", "lined"), scope='module')
def FFT2(request):
    prec = {"s": "single", "d":"double"}[request.param[-1]]
    return Line_R2C(array([N, N]), L[:-1], MPI, prec)


@pytest.fixture(params=("slabd", "slabs"), scope='module')
def FFT_C2C(request):
    prec = {"s": "single", "d":"double"}[request.param[-1]]
    return C2C(array([N, N, N]), L, MPI, prec)
    
#@profile    
def test_FFT(FFT):
    if FFT.rank == 0:
        A = random((N, N, N)).astype(FFT.float)
        if FFT.communication == 'Nyquist':
            C = empty(FFT.global_complex_shape(), dtype=FFT.complex)
            C = rfftn(A, C, axes=(0,1,2))
            C[:, :, -1] = 0  # Remove Nyquist frequency
            A = irfftn(C, A, axes=(0,1,2))
        B2 = zeros((N, N, N/2+1), dtype=FFT.complex)
        B2 = rfftn(A, B2, axes=(0,1,2))

    else:
        A = zeros((N, N, N), dtype=FFT.float)
        B2 = zeros((N, N, N/2+1), dtype=FFT.complex)

    atol, rtol = (1e-10, 1e-8) if FFT.float is float64 else (5e-7, 1e-4)
    FFT.comm.Bcast(A, root=0)
    FFT.comm.Bcast(B2, root=0)
    
    a = zeros(FFT.real_shape(), dtype=FFT.float)
    c = zeros(FFT.complex_shape(), dtype=FFT.complex)
    a[:] = A[FFT.real_local_slice()]
    c = FFT.fftn(a, c)
    #print abs((c - B2[FFT.complex_local_slice()])/c.max()).max()
    assert all(abs((c - B2[FFT.complex_local_slice()])/c.max()) < rtol)
    #assert allclose(c, B2[FFT.complex_local_slice()], rtol, atol)
    a = FFT.ifftn(c, a)
    #print abs((a - A[FFT.real_local_slice()])/a.max()).max()
    
    assert all(abs((a - A[FFT.real_local_slice()])/a.max()) < rtol)
    #assert allclose(a, A[FFT.real_local_slice()], rtol, atol)

def test_FFT2(FFT2):
    if FFT2.rank == 0:
        A = random((N, N)).astype(FFT2.float)        
        
    else:
        A = zeros((N, N), dtype=FFT2.float)

    atol, rtol = (1e-10, 1e-8) if FFT2.float is float64 else (5e-7, 1e-4)
    FFT2.comm.Bcast(A, root=0)
    a = zeros(FFT2.real_shape(), dtype=FFT2.float)
    c = zeros(FFT2.complex_shape(), dtype=FFT2.complex)
    a[:] = A[FFT2.real_local_slice()]
    c = FFT2.fft2(a, c)
    B2 = zeros(FFT2.global_complex_shape(), dtype=FFT2.complex)
    B2 = rfft2(A, B2, axes=(0,1))
    assert allclose(c, B2[FFT2.complex_local_slice()], rtol, atol)
    a = FFT2.ifft2(c, a)
    assert allclose(a, A[FFT2.real_local_slice()], rtol, atol)

def test_FFT2_padded(FFT2):
    FFT = FFT2
    N = FFT.N
    if FFT.rank == 0:
        A = random(N).astype(FFT.float)
        C = zeros((FFT.global_complex_shape()), dtype=FFT.complex)
        C = rfft2(A, C, axes=(0,1))
        
        # Eliminate Nyquist, otherwise test will fail
        C[-N[0]/2] = 0
        A = irfft2(C, A)
        
        Cp = zeros((3*N[0]/2, 3*N[1]/4+1), dtype=FFT.complex)
        Nf = N[1]/2+1
        ks = (fftfreq(N[0], 1./N[0])).astype(int)
        Cp[ks, :Nf] = C[:]
                
        Ap = zeros((3*N[0]/2, 3*N[1]/2), dtype=FFT.float)
        Ap = irfft2(Cp*1.5**2, Ap, axes=(0,1))
        
    else:
        C = zeros(FFT.global_complex_shape(), dtype=FFT.complex)
        Ap = zeros((3*N[0]/2, 3*N[1]/2), dtype=FFT.float)
        A = zeros(N, dtype=FFT.float)
        
    FFT.comm.Bcast(C, root=0)
    FFT.comm.Bcast(Ap, root=0)
    FFT.comm.Bcast(A, root=0)
    
    ae = zeros(FFT.real_shape_padded(), dtype=FFT.float)
    c = zeros(FFT.complex_shape(), dtype=FFT.complex)
    
    c[:] = C[FFT.complex_local_slice()]
    ae[:] = Ap[FFT.real_local_slice(padsize=1.5)]
    
    ap = zeros(FFT.real_shape_padded(), dtype=FFT.float)
    cp = zeros(FFT.complex_shape(), dtype=FFT.complex)
    ap = FFT.ifft2(c, ap, dealias="3/2-rule")
    
    atol, rtol = (1e-10, 1e-8) if FFT.float is float64 else (5e-7, 1e-4)
    
    #from IPython import embed; embed()
    #print np.linalg.norm(ap-ae)
    assert allclose(ap, ae, rtol, atol)
    
    cp = FFT.fft2(ap, cp, dealias="3/2-rule")    
    
    #print np.linalg.norm(abs((cp-c)/cp.max()))
    assert all(abs((cp-c)/cp.max()) < rtol)


def test_FFT_padded(FFT):
    N = FFT.N
    if FFT.rank == 0:
        A = random(N).astype(FFT.float)
        C = zeros((FFT.global_complex_shape()), dtype=FFT.complex)
        C = rfftn(A, C, axes=(0,1,2))
        
        # Eliminate Nyquist, otherwise test will fail
        C[-N[0]/2] = 0
        C[:, -N[1]/2] = 0
        if FFT.communication == 'Nyquist':
            C[:, :, -1] = 0  # Remove Nyquist frequency
        
        A = irfftn(C, A)
        
        Cp = zeros((3*N[0]/2, 3*N[1]/2, 3*N[2]/4+1), dtype=FFT.complex)
        Nf = N[2]/2+1
        ks = (fftfreq(N[1], 1./N[1])).astype(int)
        Cp[:N[0]/2, ks, :Nf] = C[:N[0]/2]
        Cp[-N[0]/2:, ks, :Nf] = C[N[0]/2:]
                
        # If Nyquist is retained then these are needed to symmetrize and pass test
        Cp[-N[0]/2] *= 0.5
        Cp[:, -N[1]/2] *= 0.5
        Cp[N[0]/2] = Cp[-N[0]/2]
        Cp[:, N[1]/2] = Cp[:, -N[1]/2]
        
        Ap = zeros((3*N[0]/2, 3*N[1]/2, 3*N[2]/2), dtype=FFT.float)
        Ap = irfftn(Cp*1.5**3, Ap, axes=(0,1,2))
        
    else:
        C = zeros(FFT.global_complex_shape(), dtype=FFT.complex)
        Ap = zeros((3*N[0]/2, 3*N[1]/2, 3*N[2]/2), dtype=FFT.float)
        A = zeros(N, dtype=FFT.float)
        
    FFT.comm.Bcast(C, root=0)
    FFT.comm.Bcast(Ap, root=0)
    FFT.comm.Bcast(A, root=0)
    
    ae = zeros(FFT.real_shape_padded(), dtype=FFT.float)
    c = zeros(FFT.complex_shape(), dtype=FFT.complex)
    
    c[:] = C[FFT.complex_local_slice()]
    ae[:] = Ap[FFT.real_local_slice(padsize=1.5)]
    
    ap = zeros(FFT.real_shape_padded(), dtype=FFT.float)
    cp = zeros(FFT.complex_shape(), dtype=FFT.complex)
    ap = FFT.ifftn(c, ap, dealias="3/2-rule")
    
    atol, rtol = (1e-10, 1e-8) if FFT.float is float64 else (5e-7, 1e-4)
    
    #print np.linalg.norm(ap-ae)
    assert allclose(ap, ae, rtol, atol)
    
    cp = FFT.fftn(ap, cp, dealias="3/2-rule")    
    
    #from IPython import embed; embed()
    #print np.linalg.norm(abs((cp-c)/cp.max()))
    assert all(abs((cp-c)/cp.max()) < rtol)

    #aa = zeros(FFT.real_shape(), dtype=FFT.float)
    #aa = FFT.ifftn(cp, aa)    
    
    #a3 = A[FFT.real_local_slice()]
    #assert allclose(aa, a3, rtol, atol)

def test_FFT_C2C(FFT_C2C):
    """Test both padded and unpadded transforms"""
    FFT = FFT_C2C
    N = FFT.N
    atol, rtol = (1e-8, 1e-8) if FFT.float is float64 else (5e-7, 1e-4)

    if FFT.rank == 0:
        # Create a reference solution using only one CPU 
        A = (random(N)+random(N)*1j).astype(FFT.complex)
        C = zeros((FFT.global_shape()), dtype=FFT.complex)
        C = fftn(A, C, axes=(0,1,2))
        
        # Copy to array padded with zeros
        Cp = zeros((3*N[0]/2, 3*N[1]/2, 3*N[2]/2), dtype=FFT.complex)
        ks = (fftfreq(N[2])*N[2]).astype(int)
        Cp[:N[0]/2, :N[1]/2, ks] = C[:N[0]/2, :N[1]/2]
        Cp[:N[0]/2, -N[1]/2:, ks] = C[:N[0]/2, N[1]/2:]
        Cp[-N[0]/2:, :N[1]/2, ks] = C[N[0]/2:, :N[1]/2]
        Cp[-N[0]/2:, -N[1]/2:, ks] = C[N[0]/2:, N[1]/2:]
        
        # Get transform of padded array
        Ap = zeros((3*N[0]/2, 3*N[1]/2, 3*N[2]/2), dtype=FFT.complex)
        Ap = ifftn(Cp*1.5**3, Ap, axes=(0,1,2))
        
    else:
        C = zeros(FFT.global_complex_shape(), dtype=FFT.complex)
        Ap = zeros((3*N[0]/2, 3*N[1]/2, 3*N[2]/2), dtype=FFT.complex)
        A = zeros(N, dtype=FFT.complex)
        
    # For testing broadcast the arrays computed on root to all CPUs
    FFT.comm.Bcast(C, root=0)
    FFT.comm.Bcast(Ap, root=0)
    FFT.comm.Bcast(A, root=0)
    
    # Get the single processor solution on local part of the solution
    ae = zeros(FFT.original_shape_padded(), dtype=FFT.complex)
    ae[:] = Ap[FFT.original_local_slice(padsize=1.5)]    
    c = zeros(FFT.transformed_shape(), dtype=FFT.complex)    
    c[:] = C[FFT.transformed_local_slice()]
    
    # Perform padded transform with MPI and assert ok
    ap = zeros(FFT.original_shape_padded(), dtype=FFT.complex)
    ap = FFT.ifftn(c, ap, dealias="3/2-rule")        
    assert allclose(ap, ae, rtol, atol)
        
    # Perform truncated transform with MPI and assert
    cp = zeros(FFT.transformed_shape(), dtype=FFT.complex)
    cp = FFT.fftn(ap, cp, dealias="3/2-rule")    
    assert all(abs(cp-c)/cp.max() < rtol)

    # Now without padding
    # Transform back to original
    aa = zeros(FFT.original_shape(), dtype=FFT.complex)
    aa = FFT.ifftn(c, aa)    
    # Verify
    a3 = A[FFT.original_local_slice()]
    assert allclose(aa, a3, rtol, atol)
    c2 = zeros(FFT.transformed_shape(), dtype=FFT.complex)    
    c2 = FFT.fftn(aa, c2)    
    # Verify
    assert all(abs(c2-c)/c2.max() < rtol)
    #assert allclose(c2, c, rtol, atol)
    
#import time
#t0 = time.time()
#test_FFT_padded(Pencil_R2C(array([N, N, N], dtype=int), L, MPI, "double", alignment="Y", communication='Swap'))
#t1 = time.time()
#test_FFT_padded(Pencil_R2C(array([N, N, N], dtype=int), L, MPI, "double", alignment="X", communication='Swap'))
#t2 = time.time()

#ty = MPI.COMM_WORLD.reduce(t1-t0, op=MPI.MIN)
#tx = MPI.COMM_WORLD.reduce(t2-t1, op=MPI.MIN)
#if MPI.COMM_WORLD.Get_rank() == 0:
    #print "Y: ", ty
    #print "X: ", tx

#test_FFT(Slab_R2C(array([N, N, N]), L, MPI, "single", communication='alltoall'))
#test_FFT(Pencil_R2C(array([N, N, N], dtype=int), L, MPI, "double", alignment="Y", communication='Alltoallw'))
#test_FFT2(Line_R2C(array([N, N]), L[:-1], MPI, "single"))
#test_FFT2_padded(Line_R2C(array([N, N]), L[:-1], MPI, "double"))
#test_FFT_padded(Slab_R2C(array([N, N, N]), L, MPI, "double", communication='Alltoallw'))
#test_FFT_padded(Pencil_R2C(array([N, N, N], dtype=int), L, MPI, "double", alignment="X", communication='Nyquist'))
#test_FFT_C2C(C2C(array([N, N, N]), L, MPI, "double"))
