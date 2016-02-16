__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2016-02-16"
__copyright__ = "Copyright (C) 2016 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from serialFFT import *
import numpy as np

class FastFourierTransform(object):
    """Class for performing FFT in 3D using MPI
    
    Slab decomposition
    
    N - NumPy array([Nx, Ny, Nz]) setting the dimensions of the real mesh
    L - NumPy array([Lx, Ly, Lz]) setting the actual size of the real mesh
    MPI - The MPI object (from mpi4py import MPI)
    precision - "single" or "double"
        
    """
    def __init__(self, N, L, MPI, precision, communication="alltoall"):
        self.N = N
        self.L = L
        assert len(L) == 3
        assert len(N) == 3
        self.Nf = N[2]/2+1 # Number of independent complex wavenumbers in z-direction 
        self.MPI = MPI
        self.comm = comm = MPI.COMM_WORLD
        self.float, self.complex, self.mpitype = float, complex, mpitype = self.types(precision)
        self.communication = communication
        self.num_processes = comm.Get_size()
        self.rank = comm.Get_rank()        
        self.Np = N / self.num_processes     
        
        # Initialize MPI work arrays globally
        self.Uc_hat  = np.empty(self.complex_shape(), dtype=complex)
        self.Uc_hatT = np.empty(self.complex_shape_T(), dtype=complex)
        self.Uc_send = self.Uc_hat.reshape((self.num_processes, self.Np[0], self.Np[1], self.Nf))
        self.U_mpi   = np.empty((self.num_processes, self.Np[0], self.Np[1], self.Nf), dtype=complex)
        
        if not self.num_processes in [2**i for i in range(int(np.log2(N[0]))+1)]:
            raise IOError("Number of cpus must be in ", [2**i for i in range(int(np.log2(N[0]))+1)])
    
    def types(self, precision):
        return {"single": (np.float32, np.complex64, self.MPI.F_FLOAT_COMPLEX),
                "double": (np.float64, np.complex128, self.MPI.F_DOUBLE_COMPLEX)}[precision]

    def real_shape(self):
        """The local shape of the real data"""
        return (self.Np[0], self.N[1], self.N[2])

    def complex_shape(self):
        """The local shape of the complex data"""
        return (self.N[0], self.Np[1], self.Nf)
    
    def complex_shape_T(self):
        """The local transposed shape of the complex data"""
        return (self.Np[0], self.N[1], self.Nf)
        
    def complex_shape_I(self):
        """A local intermediate shape of the complex data"""
        return (self.Np[0], self.num_processes, self.Np[1], self.Nf)
    
    def get_N(self):
        return self.N
    
    def get_local_mesh(self):
        # Create the physical mesh
        X = np.mgrid[self.rank*self.Np[0]:(self.rank+1)*self.Np[0], :self.N[1], :self.N[2]].astype(self.float)
        X[0] *= self.L[0]/self.N[0]
        X[1] *= self.L[1]/self.N[1]
        X[2] *= self.L[2]/self.N[2]
        return X
    
    def get_local_wavenumbermesh(self):
        kx = fftfreq(self.N[0], 1./self.N[0])
        ky = fftfreq(self.N[1], 1./self.N[1])[self.rank*self.Np[1]:(self.rank+1)*self.Np[1]]
        kz = fftfreq(self.N[2], 1./self.N[2])[:self.Nf]
        kz[-1] *= -1
        K  = np.array(np.meshgrid(kx, ky, kz, indexing='ij'), dtype=self.float)
        return K
    
    def get_scaled_local_wavenumbermesh(self):
        K = self.get_local_wavenumbermesh()
        # Scale with physical mesh size. This takes care of mapping the physical domain to a computational cube of size (2pi)**3
        Lp = 2*np.pi/self.L
        for i in range(3):
            K[i] *= Lp[i] 
        return K
    
    def get_dealias_filter(self):
        """Filter for dealiasing nonlinear convection"""
        K = self.get_local_wavenumbermesh()
        kmax = 2./3.*(self.N/2+1)
        dealias = np.array((abs(K[0]) < kmax[0])*(abs(K[1]) < kmax[1])*
                           (abs(K[2]) < kmax[2]), dtype=np.uint8)
        return dealias

    def ifftn(self, fu, u):
        """ifft in three directions using mpi.
        Need to do ifft in reversed order of fft
        """
        if self.num_processes == 1:
            u[:] = irfftn(fu, axes=(0,1,2))
            return u
        
        # Do first owned direction
        self.Uc_hat[:] = ifft(fu, axis=0)
            
        if self.communication == 'alltoall':
            # Communicate all values
            self.comm.Alltoall([self.Uc_hat, self.mpitype], [self.U_mpi, self.mpitype])
            self.Uc_hatT[:] = np.rollaxis(self.U_mpi, 1).reshape(self.Uc_hatT.shape)
        
        else:
            for i in xrange(self.num_processes):
                if not i == self.rank:
                    self.comm.Sendrecv_replace([self.Uc_send[i], self.mpitype], i, 0, i, 0)   
                self.Uc_hatT[:, i*self.Np[1]:(i+1)*self.Np[1]] = self.Uc_send[i]
            
        # Do last two directions
        u = irfft2(self.Uc_hatT, axes=(1,2))
        return u

    def fftn(self, u, fu):
        """fft in three directions using mpi
        """
        if self.num_processes == 1:
            fu[:] = rfftn(u, axes=(0,1,2))
            return fu
        
        if self.communication == 'alltoall':
            # Do 2 ffts in y-z directions on owned data
            self.Uc_hatT[:] = rfft2(u, axes=(1,2))
            
            # Transform data to align with x-direction  
            self.U_mpi[:] = np.rollaxis(self.Uc_hatT.reshape(self.Np[0], self.num_processes, self.Np[1], self.Nf), 1)
                
            # Communicate all values
            self.comm.Alltoall([self.U_mpi, self.mpitype], [fu, self.mpitype])  
        
        else:
            # Communicating intermediate result 
            ft = fu.transpose(1,0,2)
            ft[:] = rfft2(u, axes=(1,2))
            fu_send = fu.reshape((self.num_processes, self.Np[1], self.Np[1], self.Nf))
            for i in xrange(self.num_processes):
                if not i == self.rank:
                    self.comm.Sendrecv_replace([fu_send[i], self.mpitype], i, 0, i, 0)   
            fu_send[:] = fu_send.transpose(0,2,1,3)
                        
        # Do fft for last direction 
        fu[:] = fft(fu, axis=0)
        return fu
    
    #def real_shape_padded(self):
        #"""The local shape of the real data"""
        #return (3*self.Np[0]/2, 3*self.N[1]/2, 3*self.N[2]/2)
    
    #def complex_shape_padded(self):
        #return (3*self.N[0]/2, 3*self.Np[1]/2, 3*self.N[2]/4+1)

    #def complex_shape_padded_0(self):
        #return (3*self.N[0]/2, self.Np[1], self.Nf)
    
    #def complex_shape_padded_T(self):
        #"""The local shape of the transposed complex data padded in x and z directions"""
        #return (3*self.Np[0]/2, 3*self.N[1]/2, 3*self.N[2]/4+1)
    
    #def copy_to_padded(self, fu, fp):
        #fp[:self.N[0]/2] = fu[:self.N[0]/2]
        #fp[-(self.N[0]/2):] = fu[self.N[0]/2:]
        #return fp
    
    #def copy_from_padded(self, fp, fu):
        #fu[:, :self.N[1]/2] = fp[:, :self.N[1]/2, :self.Nf]
        #fu[:, self.N[1]/2:] = fp[:, -(self.N[1]/2):, :self.Nf]
        #return fu

    
    #def ifftn_padded(self, fu, u):
        #"""Inverse padded transform. 3/2-rule
        
        #fu is padded with zeros using the 3/2 rule before transforming to real space
        #"""
        #self.Upad_hat[:] = 0
        #self.Upad_hat = self.copy_to_padded(fu, self.Upad_hat) 
        #self.Uc_hat[:] = ifft(fu, axis=0)
        #self.comm.Alltoall([self.Uc_hat, self.mpitype], [self.U_mpi, self.mpitype])
        #self.Uc_hatT[:] = rollaxis(self.U_mpi, 1).reshape(self.Uc_hatT.shape)
        #self.Upad_hatT[:] = 0
        #self.Upad_hatT = self.copy_to_padded(self.Uc_hatT, self.Upad_hatT)
        #u = irfft2(1.5**2*self.Uc_hatT, axes=(1,2))
        #return u

    #def fftn_padded(self, u, fu, S):
        #"""Fast padded transform. 3/2-rule
        
        #u is of shape real_shape_padded. The output, fu, is normal complex_shape
        #"""   
        #self.Upad_hatT[:] = rfft2(u, axes=(1,2))
        #self.Uc_hatT = self.copy_from_padded(self.Upad_hatT, self.Uc_hatT)
        #self.U_mpi[:] = rollaxis(self.Uc_hatT.reshape(self.complex_shape_I()), 1)
        #self.comm.Alltoall([self.U_mpi, self.mpitype], [self.Uc_hat, self.mpitype])
        #fu = S.fst(self.Uc_hat/1.5**2, fu)
        #return fu

