__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2016-02-16"
__copyright__ = "Copyright (C) 2016 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from serialFFT import *
import numpy as np
from mpibase import work_arrays, datatypes

def transpose_x(U_send, Uc_hatT, num_processes, Np):
    # Align data in x-direction
    #for i in range(num_processes): 
    #   U_send[i] = Uc_hatT[:, i*Np[1]/2:(i+1)*Np[1]/2]
    
    sx = U_send.shape
    sy = Uc_hatT.shape
    U_send[:] = np.rollaxis(Uc_hatT[:,:-1].reshape(sy[0], num_processes, sy[0]/2), 1)
    return U_send

def transpose_y(Uc_hatT, U_recv, num_processes, Np):
    #for i in range(num_processes): 
        #Uc_hatT[:, i*Np[1]/2:(i+1)*Np[1]/2] = U_recv[i*Np[0]:(i+1)*Np[0]]
        
    sx = Uc_hatT.shape
    sy = U_recv.shape
    Uc_hatT[:, :-1] = np.rollaxis(U_recv.reshape(num_processes, sx[0], sy[1]), 1).reshape((sx[0], sx[1]-1))
    return Uc_hatT

def swap_Nq(fft_y, fu, fft_x, N):
    f = fu[:, 0].copy()        
    fft_x[0] = f[0].real
    fft_x[1:N[0]/2] = 0.5*(f[1:N[0]/2] + np.conj(f[:N[0]/2:-1]))
    fft_x[N[0]/2] = f[N[0]/2].real        
    fu[:N[0]/2+1, 0] = fft_x[:N[0]/2+1]        
    fu[N[0]/2+1:, 0] = np.conj(fft_x[(N[0]/2-1):0:-1])
    
    fft_y[0] = f[0].imag
    fft_y[1:N[0]/2] = -0.5*1j*(f[1:N[0]/2] - np.conj(f[:N[0]/2:-1]))
    fft_y[N[0]/2] = f[N[0]/2].imag
    
    fft_y[N[0]/2+1:] = np.conj(fft_y[(N[0]/2-1):0:-1])
    return fft_y

class FastFourierTransform(object):
    """Class for performing FFT in 2D using MPI
    
    Slab decomposition
    
    N - NumPy array([Nx, Ny]) Number of nodes for the real mesh
    L - NumPy array([Lx, Ly]) The actual size of the real mesh
    MPI - The MPI object (from mpi4py import MPI)
    precision - "single" or "double"
        
    """
    
    def __init__(self, N, L, MPI, precision):
        self.N = N         # The global size of the problem
        self.L = L
        assert len(L) == 2
        assert len(N) == 2
        self.MPI = MPI
        self.comm = comm = MPI.COMM_WORLD
        self.float, self.complex, self.mpitype = float, complex, mpitype = datatypes(precision)
        self.num_processes = comm.Get_size()
        self.rank = comm.Get_rank()
        # Each cpu gets ownership of Np indices
        self.Np = N / self.num_processes
        self.Nf = N[1]/2+1
        self.Npf = self.Np[1]/2+1 if self.rank+1 == self.num_processes else self.Np[1]/2
        self.dealias = None

        self.U_recv = empty((self.N[0], self.Np[1]/2), dtype=complex)
        self.fft_y = empty(N[0], dtype=complex)
        self.fft_x = empty(N[0], dtype=complex)
        self.plane_recv = empty(self.Np[0], dtype=complex)
        self.Uc_hat = empty((N[0], self.Npf), dtype=complex)
        self.Uc_hatT = empty((self.Np[0], self.Nf), dtype=complex)
        self.U_send = empty((self.num_processes, self.Np[0], self.Np[1]/2), dtype=complex)
        self.U_sendr = self.U_send.reshape((N[0], self.Np[1]/2))

    def real_shape(self):
        """The local shape of the real data"""
        return (self.Np[0], self.N[1])

    def complex_shape(self):
        """The local shape of the complex data"""
        return (self.N[0], self.Npf)

    def real_local_slice(self):
        return (slice(self.rank*self.Np[0], (self.rank+1)*self.Np[0], 1),
                slice(0, self.N[1]))
    
    def complex_local_slice(self):
        return (slice(0, self.N[0]), 
                slice(self.rank*self.Np[1]/2, self.rank*self.Np[1]/2+self.Npf, 1))
        
    def get_N(self):
        return self.N

    def get_local_mesh(self):
        # Create the mesh
        X = np.mgrid[self.rank*self.Np[0]:(self.rank+1)*self.Np[0], :self.N[1]].astype(self.float)
        X[0] *= self.L[0]/self.N[0]
        X[1] *= self.L[1]/self.N[1]
        return X
    
    def get_local_wavenumbermesh(self):
        kx = fftfreq(self.N[0], 1./self.N[0])
        ky = fftfreq(self.N[1], 1./self.N[1])[:self.Nf]
        ky[-1] *= -1
        K = np.array(np.meshgrid(kx, ky[self.rank*self.Np[1]/2:(self.rank*self.Np[1]/2+self.Npf)], indexing='ij'), dtype=self.float)
        return K
    
    def get_scaled_local_wavenumbermesh(self):
        K = self.get_local_wavenumbermesh()
        Lp = 2*np.pi/self.L
        K[0] *= Lp[0]
        K[1] *= Lp[1]
        return K
    
    def get_dealias_filter(self):
        """Filter for dealiasing nonlinear convection"""
        K = self.get_local_wavenumbermesh()
        kmax = 2./3.*(self.N/2+1)
        dealias = np.array((abs(K[0]) < kmax[0])*(abs(K[1]) < kmax[1]), dtype=np.uint8)
        return dealias

    def fft2(self, u, fu, dealias=None):
        assert dealias in ('2/3-rule', 'None', None)
        
        if self.num_processes == 1:
            fu[:] = rfft2(u, axes=(0,1))
            return fu    
        
        self.Uc_hatT[:] = rfft(u, axis=1)
        self.Uc_hatT[:, 0] += 1j*self.Uc_hatT[:, -1]
        
        self.U_send = transpose_x(self.U_send, self.Uc_hatT, self.num_processes, self.Np)
                
        # Communicate all values
        self.comm.Alltoall([self.U_send, self.mpitype], [self.U_recv, self.mpitype])
        
        fu[:, :self.Np[1]/2] = fft(self.U_recv, axis=0)
        
        # Handle Nyquist frequency
        if self.rank == 0:        
            self.fft_y = swap_Nq(self.fft_y, fu, self.fft_x, self.N)
            self.comm.Send([self.fft_y, self.mpitype], dest=self.num_processes-1, tag=77)
            
        elif self.rank == self.num_processes-1:
            self.comm.Recv([self.fft_y, self.mpitype], source=0, tag=77)
            fu[:, -1] = self.fft_y 
            
        return fu

    def ifft2(self, fu, u, dealias=None):
        assert dealias in ('2/3-rule', 'None', None)
        
        if dealias == '2/3-rule':
            if self.dealias is None:
                self.dealias = self.get_dealias_filter()
            fu *= self.dealias
        
        if self.num_processes == 1:
            u[:] = irfft2(fu, axes=(0,1))
            return u

        self.Uc_hat[:] = ifft(fu, axis=0)    
        self.U_sendr[:] = self.Uc_hat[:, :self.Np[1]/2]

        self.comm.Alltoall([self.U_send, self.mpitype], [self.U_recv, self.mpitype])

        self.Uc_hatT = transpose_y(self.Uc_hatT, self.U_recv, self.num_processes, self.Np)
        
        if self.rank == self.num_processes-1:
            self.fft_y[:] = self.Uc_hat[:, -1]

        self.comm.Scatter(self.fft_y, self.plane_recv, root=self.num_processes-1)
        self.Uc_hatT[:, -1] = self.plane_recv
        
        u[:] = irfft(self.Uc_hatT, axis=1)
        return u

    def get_workarray(self, a, i=0):
        if isinstance(a, np.ndarray):
            shape = a.shape
            dtype = a.dtype
            
        elif isinstance(a, tuple):
            assert len(a) == 2
            shape, dtype = a
            
        else:
            raise TypeError("Wrong type for get_workarray")
        
        a = work_arrays[(shape, dtype, i)]
        a[:] = 0
        return a
