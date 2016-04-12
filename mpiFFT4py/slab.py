__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2016-02-16"
__copyright__ = "Copyright (C) 2016 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from serialFFT import *
import numpy as np

class work_arrays(dict):
    
    def __missing__(self, key):
        shape, dtype, i = key
        a = zeros(shape, dtype=dtype)
        self[key] = a
        return self[key]

_work_arrays = work_arrays()

class FastFourierTransform(object):
    """Class for performing FFT in 3D using MPI
    
    Slab decomposition
        
    N - NumPy array([Nx, Ny, Nz]) Number of nodes for the real mesh
    L - NumPy array([Lx, Ly, Lz]) The actual size of the real mesh
    MPI - The MPI object (from mpi4py import MPI)
    precision - "single" or "double"
    
    The transform is real to complex
    
    """
    def __init__(self, N, L, MPI, precision, communication="alltoall"):
        self.N = N
        assert len(L) == 3
        assert len(N) == 3
        self.Nf = N[2]/2+1 # Number of independent complex wavenumbers in z-direction 
        self.MPI = MPI
        self.comm = comm = MPI.COMM_WORLD
        self.float, self.complex, self.mpitype = self.types(precision)
        self.communication = communication
        self.num_processes = comm.Get_size()
        self.rank = comm.Get_rank()        
        self.Np = N / self.num_processes     
        self.L = L.astype(self.float)
        self.Uc_hat = None
        self.Upad_hat = None
        self.dealias = None
        if not self.num_processes in [2**i for i in range(int(np.log2(N[0]))+1)]:
            raise IOError("Number of cpus must be in ", [2**i for i in range(int(np.log2(N[0]))+1)])
        
    def init_work_arrays(self, padded=False):
        
        if padded:
            if self.Upad_hat is None:
                self.Upad_hat = np.zeros(self.complex_shape_padded_0(), dtype=self.complex)
                self.Upad_hat0 = np.zeros(self.complex_shape_padded_0(), dtype=self.complex)
                self.U_mpi = np.zeros(self.complex_shape_padded_0_I(), dtype=self.complex)
                self.Upad_hat1 = np.zeros(self.complex_shape_padded_1(), dtype=self.complex)
                self.Upad_hat2 = np.zeros(self.complex_shape_padded_2(), dtype=self.complex)
                self.Upad_hat3 = np.zeros(self.complex_shape_padded_3(), dtype=self.complex)
            else:
                self.Upad_hat[:] = 0
                self.Upad_hat0[:] = 0
                self.U_mpi[:] = 0
                self.Upad_hat1[:] = 0
                self.Upad_hat2[:] = 0
                self.Upad_hat3[:] = 0
            
        else:
            # Initialize regular MPI work arrays
            if self.Uc_hat is None:
                self.Uc_hat  = np.empty(self.complex_shape(), dtype=self.complex)
                self.Uc_hatT = np.empty(self.complex_shape_T(), dtype=self.complex)
                self.Uc_send = self.Uc_hat.reshape((self.num_processes, self.Np[0], self.Np[1], self.Nf))
                self.Uc_mpi   = np.empty((self.num_processes, self.Np[0], self.Np[1], self.Nf), dtype=self.complex)
            else:
                self.Uc_hat[:]  = 0
                self.Uc_hatT[:] = 0
                self.Uc_send[:] = 0
                self.Uc_mpi[:]   = 0
        
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
    
    def global_real_shape(self):
        """Global size of problem in real physical space"""
        return (self.N[0], self.N[1], self.N[2])
    
    def global_complex_shape(self):
        """Global size of problem in complex wavenumber space"""
        return (self.N[0], self.N[1], self.Nf)

    def global_complex_shape_padded(self):
        """Global size of problem in complex wavenumber space"""
        return (3*self.N[0]/2, 3*self.N[1]/2, 3*self.N[2]/4+1)
    
    def real_local_slice(self, padded=False):
        if padded:
            return (slice(3*self.rank*self.Np[0]/2, 3*(self.rank+1)*self.Np[0]/2, 1),
                    slice(0, 3*self.N[1]/2, 1), 
                    slice(0, 3*self.N[2]/2, 1))
        else:
            return (slice(self.rank*self.Np[0], (self.rank+1)*self.Np[0], 1),
                    slice(0, self.N[1], 1), 
                    slice(0, self.N[2], 1))
    
    def complex_local_slice(self):
        return (slice(0, self.N[0], 1),
                slice(self.rank*self.Np[1], (self.rank+1)*self.Np[1], 1),
                slice(0, self.Nf, 1))

    def complex_local_wavenumbers(self):
        return (fftfreq(self.N[0], 1./self.N[0]),
                fftfreq(self.N[1], 1./self.N[1])[self.rank*self.Np[1]:(self.rank+1)*self.Np[1]],
                rfftfreq(self.N[2], 1./self.N[2])[:self.Nf])
    
    def get_local_mesh(self):
        # Create the physical mesh
        X = np.mgrid[self.rank*self.Np[0]:(self.rank+1)*self.Np[0], :self.N[1], :self.N[2]].astype(self.float)
        X[0] *= self.L[0]/self.N[0]
        X[1] *= self.L[1]/self.N[1]
        X[2] *= self.L[2]/self.N[2]
        return X
    
    def get_local_wavenumbermesh(self):
        kx, ky, kz = self.complex_local_wavenumbers()
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

    def ifftn(self, fu, u, dealias=None):
        """ifft in three directions using mpi.
        Need to do ifft in reversed order of fft

        dealias = "3/2-rule"
            - Padded transform with 3/2-rule. fu is padded with zeros
              before transforming to real space of shape real_shape_padded()
            - u real_shape_padded()
        
        dealias = "2/3-rule"
            - Transform is using 2/3-rule, i.e., frequencies higher than
              2/3*N are set to zero before transforming
            - u real_shape()
              
        dealias = None
            - Regular transform
            - u real_shape()
            
        fu is of shape complex_shape()
        
        """

        if self.num_processes == 1:
            if not dealias == '3/2-rule':
                if dealias == '2/3-rule':
                    if self.dealias is None:
                        self.dealias = self.get_dealias_filter()
                    fu *= self.dealias
                
                u[:] = irfftn(fu, axes=(0,1,2))
            
            else:
                assert u.shape == self.real_shape_padded()

                # First create padded complex array and then perform irfftn
                fu_padded = zeros(self.global_complex_shape_padded(), dtype=self.complex)
                ks = (fftfreq(self.N[1])*self.N[1]).astype(int)
                fu_padded[:self.N[0]/2, ks, :self.Nf] = fu[:self.N[0]/2]
                fu_padded[-self.N[0]/2:, ks, :self.Nf] = fu[self.N[0]/2:]
                
                ## Current transform is only exactly reversible if periodic transforms are made symmetric
                ## However, this leads to more aliasing and as such the non-symmetrical padding is used
                #fu_padded[:, -self.N[1]/2] *= 0.5
                #fu_padded[-self.N[0]/2] *= 0.5
                #fu_padded[self.N[0]/2] = fu_padded[-self.N[0]/2]
                #fu_padded[:, self.N[1]/2] = fu_padded[:, -self.N[0]/2]
                
                u[:] = irfftn(fu_padded*1.5**3, axes=(0,1,2))
            return u
        
        self.init_work_arrays(dealias == '3/2-rule')
        
        if not dealias == '3/2-rule':
            if dealias == '2/3-rule':
                if self.dealias is None:
                    self.dealias = self.get_dealias_filter()
                fu *= self.dealias
    
            # Do first owned direction
            self.Uc_hat[:] = ifft(fu, axis=0)
                
            if self.communication == 'alltoall':
                # Communicate all values
                self.comm.Alltoall([self.Uc_hat, self.mpitype], [self.Uc_mpi, self.mpitype])
                self.Uc_hatT[:] = np.rollaxis(self.Uc_mpi, 1).reshape(self.Uc_hatT.shape)
            
            else:
                for i in xrange(self.num_processes):
                    if not i == self.rank:
                        self.comm.Sendrecv_replace([self.Uc_send[i], self.mpitype], i, 0, i, 0)   
                    self.Uc_hatT[:, i*self.Np[1]:(i+1)*self.Np[1]] = self.Uc_send[i]
                
            # Do last two directions
            u[:] = irfft2(self.Uc_hatT, axes=(1,2))

        else:
            self.Upad_hat = self.copy_to_padded_x(fu, self.Upad_hat)
            self.Upad_hat[:] = ifft(self.Upad_hat*1.5, axis=0)        
            self.comm.Alltoall([self.Upad_hat, self.mpitype], [self.U_mpi, self.mpitype])
            self.Upad_hat1[:] = np.rollaxis(self.U_mpi, 1).reshape(self.Upad_hat1.shape)
            self.Upad_hat2 = self.copy_to_padded_y(self.Upad_hat1, self.Upad_hat2)
            self.Upad_hat2[:] = ifft(self.Upad_hat2*1.5, axis=1)
            self.Upad_hat3 = self.copy_to_padded_z(self.Upad_hat2, self.Upad_hat3)
            u[:] = irfft(self.Upad_hat3*1.5, axis=2)
        return u


    def fftn(self, u, fu, dealias=None):
        """fft in three directions using mpi
        
        u is real an either of shape real_shape() or real_shape_padded()
        
        fu is complex and of shape complex_shape()
        
        """
        if self.num_processes == 1:
            if not dealias == '3/2-rule':
                assert u.shape == self.real_shape()
                
                fu[:] = rfftn(u, axes=(0,1,2))
            
            else:
                assert u.shape == self.real_shape_padded()
                
                fu_padded = zeros(self.global_complex_shape_padded(), dtype=self.complex)
                fu_padded[:] = rfftn(u/1.5**3, axes=(0,1,2))
                
                # Copy with truncation
                ks = (fftfreq(self.N[1])*self.N[1]).astype(int)
                fu[:self.N[0]/2] = fu_padded[:self.N[0]/2, ks, :self.Nf] 
                fu[self.N[0]/2:] = fu_padded[-self.N[0]/2:, ks, :self.Nf] 
                
                ## Modify for symmetric padding
                #fu[:, -self.N[1]/2] *= 2
                #fu[self.N[0]/2] *= 2                
                                
            return fu
        
        self.init_work_arrays(dealias == '3/2-rule')
        
        if not dealias == '3/2-rule':
            if self.communication == 'alltoall':
                # Do 2 ffts in y-z directions on owned data
                self.Uc_hatT[:] = rfft2(u, axes=(1,2))
                
                # Transform data to align with x-direction  
                self.Uc_mpi[:] = np.rollaxis(self.Uc_hatT.reshape(self.Np[0], self.num_processes, self.Np[1], self.Nf), 1)
                    
                # Communicate all values
                self.comm.Alltoall([self.Uc_mpi, self.mpitype], [fu, self.mpitype])  
            
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
        
        else:
            # Do ffts in y and z directions
            self.Upad_hat3[:] = rfft2(u/1.5**2, axes=(1,2))        
            
            # Copy with truncation 
            self.Upad_hat1 = self.copy_from_padded(self.Upad_hat3, self.Upad_hat1)
            
            # Transpose and commuincate data
            self.U_mpi[:] = np.rollaxis(self.Upad_hat1.reshape(self.complex_shape_padded_I()), 1)
            self.comm.Alltoall([self.U_mpi, self.mpitype], [self.Upad_hat0, self.mpitype])
            
            # Perform fft of data in x-direction
            self.Upad_hat[:] = fft(self.Upad_hat0/1.5, axis=0)
            
            # Truncate to original complex shape
            fu[:self.N[0]/2] = self.Upad_hat[:self.N[0]/2]
            fu[self.N[0]/2:] = self.Upad_hat[-self.N[0]/2:]
        
        return fu
    
    def real_shape_padded(self):
        """The local shape of the real data"""
        return (3*self.Np[0]/2, 3*self.N[1]/2, 3*self.N[2]/2)
    
    def complex_shape_padded_0(self):
        """Padding in x-direction"""
        return (3*self.N[0]/2, self.Np[1], self.Nf)

    def complex_shape_padded_0_I(self):
        """Padding in x-direction"""
        return (self.num_processes, 3*self.Np[0]/2, self.Np[1], self.Nf)

    def complex_shape_padded_1(self):
        """Transpose complex_shape_padded_0"""
        return (3*self.Np[0]/2, self.N[1], self.Nf)
    
    def complex_shape_padded_2(self):
        """Padding in x and y-directions"""
        return (3*self.Np[0]/2, 3*self.N[1]/2, self.Nf)
    
    def complex_shape_padded_3(self):
        """Padding in all directions. 
        ifft of this shape leads to real_shape_padded"""
        return (3*self.Np[0]/2, 3*self.N[1]/2, 3*self.N[2]/4+1)

    def complex_shape_padded_I(self):
        """A local intermediate shape of the complex data"""
        return (3*self.Np[0]/2, self.num_processes, self.Np[1], self.Nf)
    
    def copy_to_padded_x(self, fu, fp):
        fp[:self.N[0]/2] = fu[:self.N[0]/2]
        fp[-(self.N[0]/2):] = fu[self.N[0]/2:]
        return fp

    def copy_to_padded_y(self, fu, fp):
        fp[:, :self.N[1]/2] = fu[:, :self.N[1]/2]
        fp[:, -(self.N[1]/2):] = fu[:, self.N[1]/2:]
        return fp
    
    def copy_to_padded_z(self, fu, fp):
        fp[:, :, :self.Nf] = fu[:]
        return fp
    
    def copy_from_padded(self, fp, fu):
        fu[:, :self.N[1]/2] = fp[:, :self.N[1]/2, :self.Nf]
        fu[:, self.N[1]/2:] = fp[:, -(self.N[1]/2):, :self.Nf]
        return fu
    
    def get_workarray(self, a, i=0):
        if isinstance(a, ndarray):
            shape = a.shape
            dtype = a.dtype
            
        elif isinstance(a, tuple):
            assert len(a) == 2
            shape, dtype = a
            
        else:
            raise TypeError("Wrong type for get_workarray")
        
        a = _work_arrays[(shape, dtype, i)]
        a[:] = 0
        return a
