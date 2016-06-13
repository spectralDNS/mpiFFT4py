__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2016-02-16"
__copyright__ = "Copyright (C) 2016 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from serialFFT import *
import numpy as np
from mpibase import work_arrays, datatypes
from mpiFFT4py.cython.maths import dealias_filter

class FastFourierTransform(object):
    """Class for performing FFT in 3D using MPI
    
    Slab decomposition
        
    N - NumPy array([Nx, Ny, Nz]) Number of nodes for the real mesh
    L - NumPy array([Lx, Ly, Lz]) The actual size of the real mesh
    MPI - The MPI object (from mpi4py import MPI)
    precision - "single" or "double"
    
    The transform is real to complex
    
    """
    def __init__(self, N, L, MPI, precision, communication="alltoall", padsize=1.5, threads=1):
        assert len(L) == 3
        assert len(N) == 3
        self.N = N
        self.Nf = N[2]/2+1 # Number of independent complex wavenumbers in z-direction 
        self.Nfp = int(padsize*N[2]/2+1) # Number of independent complex wavenumbers in z-direction for padded array        
        self.MPI = MPI
        self.comm = comm = MPI.COMM_WORLD
        self.float, self.complex, self.mpitype = datatypes(precision)
        self.communication = communication
        self.num_processes = comm.Get_size()
        self.rank = comm.Get_rank()        
        self.Np = N / self.num_processes
        self.L = L.astype(self.float)
        self.dealias = np.zeros(0)
        self.padsize = padsize
        self.threads = threads
        self.transform = 'r2c/c2r'
        self.work_arrays = work_arrays()
        self.ks = (fftfreq(N[1])*N[1]).astype(int)
        if not self.num_processes in [2**i for i in range(int(np.log2(N[0]))+1)]:
            raise IOError("Number of cpus must be in ", [2**i for i in range(int(np.log2(N[0]))+1)])
        
    def real_shape(self):
        """The local shape of the real data"""
        return (self.Np[0], self.N[1], self.N[2])

    def complex_shape(self):
        """The local shape of the complex data"""
        return (self.N[0], self.Np[1], self.Nf)
    
    def complex_shape_T(self):
        """The local transposed shape of the complex data"""
        return (self.Np[0], self.N[1], self.Nf)
        
    def global_real_shape(self):
        """Global size of problem in real physical space"""
        return (self.N[0], self.N[1], self.N[2])
    
    def global_complex_shape(self):
        """Global size of problem in complex wavenumber space"""
        return (self.N[0], self.N[1], self.Nf)

    def global_complex_shape_padded(self):
        """Global size of problem in complex wavenumber space"""
        return (int(self.padsize*self.N[0]), int(self.padsize*self.N[1]), self.Nfp)
    
    def work_shape(self, dealias):
        """Shape of work arrays used in convection with dealiasing. Different shape whether or not padding is involved"""
        if dealias == '3/2-rule':
            return self.real_shape_padded()
        
        else:
            return self.real_shape()
    
    def real_local_slice(self, padded=False):
        if padded:
            return (slice(int(self.padsize*self.rank*self.Np[0]), int(self.padsize*(self.rank+1)*self.Np[0]), 1),
                    slice(0, int(self.padsize*self.N[1]), 1), 
                    slice(0, int(self.padsize*self.N[2]), 1))
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
                rfftfreq(self.N[2], 1./self.N[2]))
    
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
    #@profile
    def ifftn(self, fu, u, dealias=None):
        """ifft in three directions using mpi.
        Need to do ifft in reversed order of fft

        dealias = "3/2-rule"
            - Padded transform with 3/2-rule. fu is padded with zeros
              before transforming to real space of shape real_shape_padded()
            - u is of real_shape_padded()
        
        dealias = "2/3-rule"
            - Transform is using 2/3-rule, i.e., frequencies higher than
              2/3*N are set to zero before transforming
            - u is of real_shape()
              
        dealias = None
            - Regular transform
            - u is of real_shape()
            
        fu is of complex_shape()
        
        """
        assert dealias in ('3/2-rule', '2/3-rule', 'None', None)

        if dealias == '2/3-rule' and self.dealias.shape == (0,):
            self.dealias = self.get_dealias_filter()

        if dealias == '2/3-rule':
            fu = dealias_filter(fu, self.dealias)
            #fu *= self.dealias

        if self.num_processes == 1:
            if not dealias == '3/2-rule':                
                u = irfftn(fu, u, axes=(0,1,2), threads=self.threads)
            
            else:
                assert u.shape == self.real_shape_padded()

                # First create padded complex array and then perform irfftn
                fu_padded = self.work_arrays[(self.global_complex_shape_padded(), self.complex, 0, False)]
                fu_padded[:self.N[0]/2, self.ks, :self.Nf] = fu[:self.N[0]/2]
                fu_padded[-self.N[0]/2:, self.ks, :self.Nf] = fu[self.N[0]/2:]
                
                ## Current transform is only exactly reversible if periodic transforms are made symmetric
                ## However, this seems to lead to more aliasing and as such the non-symmetrical padding is used
                #fu_padded[:, -self.N[1]/2] *= 0.5
                #fu_padded[-self.N[0]/2] *= 0.5
                #fu_padded[self.N[0]/2] = fu_padded[-self.N[0]/2]
                #fu_padded[:, self.N[1]/2] = fu_padded[:, -self.N[1]/2]
                
                u[:] = irfftn(fu_padded*self.padsize**3, overwrite_input=True, axes=(0,1,2), threads=self.threads)
            return u
        
        if not dealias == '3/2-rule':
            # Intermediate work arrays required for transform
            Uc_hat  = self.work_arrays[(self.complex_shape(), self.complex, 0, False)]
            #Uc_mpi  = self.work_arrays[((self.num_processes, self.Np[0], self.Np[1], self.Nf), self.complex, 0)]
            Uc_hatT = self.work_arrays[(self.complex_shape_T(), self.complex, 0, False)]
            
            # Do first owned direction
            Uc_hat = ifft(fu, Uc_hat, axis=0, threads=self.threads)
                
            if self.communication == 'alltoall':
                # Communicate all values
                #self.comm.Alltoall([Uc_hat, self.mpitype], [Uc_mpi, self.mpitype])
                #Uc_hatT[:] = np.rollaxis(Uc_mpi, 1).reshape(Uc_hatT.shape)
                self.comm.Alltoall(self.MPI.IN_PLACE, [Uc_hat, self.mpitype])
                Uc_hatT[:] = np.rollaxis(Uc_hat.reshape((self.num_processes, self.Np[0], self.Np[1], self.Nf)), 1).reshape(Uc_hatT.shape)
            
            else:
                Uc_send = Uc_hat.reshape((self.num_processes, self.Np[0], self.Np[1], self.Nf))
                for i in xrange(self.num_processes):
                    if not i == self.rank:
                        self.comm.Sendrecv_replace([Uc_send[i], self.mpitype], i, 0, i, 0)   
                    Uc_hatT[:, i*self.Np[1]:(i+1)*self.Np[1]] = Uc_send[i]
                
            # Do last two directions
            u[:] = irfft2(Uc_hatT, overwrite_input=True, axes=(1,2), threads=self.threads)

        else:
            assert self.num_processes <= self.N[0]/2, "Number of processors cannot be larger than N[0]/2 for 3/2-rule"            
            
            # Intermediate work arrays required for transform
            Upad_hat  = self.work_arrays[(self.complex_shape_padded_0(), self.complex, 0, False)]
            Upad_hat1 = self.work_arrays[(self.complex_shape_padded_1(), self.complex, 0, False)]
            Upad_hat2 = self.work_arrays[(self.complex_shape_padded_2(), self.complex, 0, False)]
            Upad_hat3 = self.work_arrays[(self.complex_shape_padded_3(), self.complex, 0, False)]
            
            # Expand in x-direction and perform ifft
            Upad_hat = self.copy_to_padded_x(fu, Upad_hat)
            Upad_hat[:] = ifft(Upad_hat, axis=0, threads=self.threads)
            
            # Communicate to distribute first dimension (like Fig. 2b but padded in x-dir)
            self.comm.Alltoall(self.MPI.IN_PLACE, [Upad_hat, self.mpitype])
            U_mpi = Upad_hat.reshape(self.complex_shape_padded_0_I())
            
            # Transpose data and pad in y-direction before doing ifft. Now data is padded in x and y 
            Upad_hat1[:] = np.rollaxis(U_mpi, 1).reshape(Upad_hat1.shape)
            Upad_hat2 = self.copy_to_padded_y(Upad_hat1, Upad_hat2)
            Upad_hat2[:] = ifft(Upad_hat2, axis=1, threads=self.threads)
            
            # pad in z-direction and perform final irfft
            Upad_hat3 = self.copy_to_padded_z(Upad_hat2, Upad_hat3)
            u[:] = irfft(Upad_hat3*self.padsize**3, overwrite_input=True, axis=2, threads=self.threads)
            
        return u

    #@profile
    def fftn(self, u, fu, dealias=None):
        """fft in three directions using mpi
        
        dealias = "3/2-rule"
            - Truncated transform with 3/2-rule. The transfored fu is truncated
              when copied to complex space of complex_shape()
            - fu is of complex_shape()
            - u is of real_shape_padded()
        
        dealias = "2/3-rule"
            - Regular transform 
            - fu is of complex_shape()
            - u is of real_shape()
              
        dealias = None
            - Regular transform
            - fu is of complex_shape()
            - u is of real_shape()
            
        """
        assert dealias in ('3/2-rule', '2/3-rule', 'None', None)
        
        if self.num_processes == 1:
            if not dealias == '3/2-rule':
                assert u.shape == self.real_shape()
                
                fu = rfftn(u, fu, axes=(0,1,2), threads=self.threads)
            
            else:
                assert u.shape == self.real_shape_padded()
                
                fu_padded = self.work_arrays[(self.global_complex_shape_padded(), self.complex, 0, False)]
                fu_padded[:] = rfftn(u/self.padsize**3, overwrite_input=True, axes=(0,1,2))
                
                # Copy with truncation
                fu[:self.N[0]/2] = fu_padded[:self.N[0]/2, self.ks, :self.Nf] 
                fu[self.N[0]/2:] = fu_padded[-self.N[0]/2:, self.ks, :self.Nf] 
                
                ## Modify for symmetric padding
                #fu[:, -self.N[1]/2] *= 2
                #fu[self.N[0]/2] *= 2                
                                
            return fu
        
        if not dealias == '3/2-rule':
            if self.communication == 'alltoall':     
                # Intermediate work arrays required for transform
                Uc_hatT = self.work_arrays[(self.complex_shape_T(), self.complex, 0, False)]
                
                # Do 2 ffts in y-z directions on owned data
                Uc_hatT = rfft2(u, Uc_hatT, axes=(1,2), threads=self.threads)
                
                # Transform data to align with x-direction  
                fu[:] = np.rollaxis(Uc_hatT.reshape(self.Np[0], self.num_processes, self.Np[1], self.Nf), 1).reshape(fu.shape)
                    
                # Communicate all values
                self.comm.Alltoall(self.MPI.IN_PLACE, [fu, self.mpitype])  # Note to self. In place is possible, should test for efficiency
            
            else:
                # Communicating intermediate result 
                ft = fu.transpose(1,0,2)
                ft = rfft2(u, ft, axes=(1,2), threads=self.threads)
                fu_send = fu.reshape((self.num_processes, self.Np[1], self.Np[1], self.Nf))
                for i in xrange(self.num_processes):
                    if not i == self.rank:
                        self.comm.Sendrecv_replace([fu_send[i], self.mpitype], i, 0, i, 0)   
                fu_send[:] = fu_send.transpose(0,2,1,3)
                            
            # Do fft for last direction 
            fu[:] = fft(fu, axis=0, threads=self.threads)
        
        else:
            assert self.num_processes <= self.N[0]/2, "Number of processors cannot be larger than N[0]/2 for 3/2-rule"
            assert u.shape == self.real_shape_padded()
            
            # Intermediate work arrays required for transform
            Upad_hat  = self.work_arrays[(self.complex_shape_padded_0(), self.complex, 0, False)]
            Upad_hat0 = self.work_arrays[(self.complex_shape_padded_0(), self.complex, 1, False)]
            Upad_hat1 = self.work_arrays[(self.complex_shape_padded_1(), self.complex, 0, False)]
            Upad_hat2 = self.work_arrays[(self.complex_shape_padded_2(), self.complex, 0, False)]
            Upad_hat3 = self.work_arrays[(self.complex_shape_padded_3(), self.complex, 0, False)]
            
            # Do ffts in the padded y and z directions
            Upad_hat3 = rfft2(u/self.padsize**2, Upad_hat3, axes=(1,2), threads=self.threads)        
            
            # Copy with truncation 
            Upad_hat1 = self.copy_from_padded(Upad_hat3, Upad_hat1)
            
            # Transpose and commuincate data
            Upad_hat0[:] = np.rollaxis(Upad_hat1.reshape(self.complex_shape_padded_I()), 1).reshape(Upad_hat0.shape)
            self.comm.Alltoall(self.MPI.IN_PLACE, [Upad_hat0, self.mpitype])
            
            # Perform fft of data in x-direction
            Upad_hat[:] = fft(Upad_hat0/self.padsize, overwrite_input=True, axis=0, threads=self.threads)
            
            # Truncate to original complex shape
            fu[:self.N[0]/2] = Upad_hat[:self.N[0]/2]
            fu[self.N[0]/2:] = Upad_hat[-self.N[0]/2:]
        
        return fu
    
    def real_shape_padded(self):
        """The local shape of the real data"""
        return (int(self.padsize*self.Np[0]), int(self.padsize*self.N[1]), int(self.padsize*self.N[2]))
    
    def complex_shape_padded_0(self):
        """Padding in x-direction"""
        return (int(self.padsize*self.N[0]), self.Np[1], self.Nf)

    def complex_shape_padded_0_I(self):
        """Padding in x-direction - reshaped for MPI communications"""
        return (self.num_processes, int(self.padsize*self.Np[0]), self.Np[1], self.Nf)

    def complex_shape_padded_1(self):
        """Transpose of complex_shape_padded_0"""
        return (int(self.padsize*self.Np[0]), self.N[1], self.Nf)
    
    def complex_shape_padded_2(self):
        """Padding in x and y-directions"""
        return (int(self.padsize*self.Np[0]), int(self.padsize*self.N[1]), self.Nf)
    
    def complex_shape_padded_3(self):
        """Padding in all directions. 
        ifft of this shape leads to real_shape_padded"""
        return (int(self.padsize*self.Np[0]), int(self.padsize*self.N[1]), self.Nfp)

    def complex_shape_padded_I(self):
        """A local intermediate shape of the complex data"""
        return (int(self.padsize*self.Np[0]), self.num_processes, self.Np[1], self.Nf)
    
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

class c2c(FastFourierTransform):
    
    def __init__(self, N, L, MPI, precision, communication="alltoall", padsize=1.5, threads=1):
        FastFourierTransform.__init__(self, N, L, MPI, precision, 
                                      communication=communication, 
                                      padsize=padsize, threads=threads)
        # Reuse all shapes from r2c transform FastFourierTransform simply by resizing the final complex z-dimension:
        self.Nf = N[2]      
        self.Nfp = int(self.padsize*self.N[2]) # Number of independent complex wavenumbers in z-direction for padded array
        self.transform = 'c2c/c2c'
        
        # Rename since there's no real space 
        self.original_shape_padded = self.real_shape_padded
        self.original_shape = self.real_shape
        self.transformed_shape = self.complex_shape
        self.global_shape = self.global_complex_shape
        self.original_local_slice = self.real_local_slice
        self.transformed_local_slice = self.complex_local_slice
        self.ks = (fftfreq(N[2])*N[2]).astype(int)
        
    def transformed_local_wavenumbers(self):
        return (fftfreq(self.N[0], 1./self.N[0]),
                fftfreq(self.N[1], 1./self.N[1])[self.rank*self.Np[1]:(self.rank+1)*self.Np[1]],
                fftfreq(self.N[2], 1./self.N[2]))

    def ifftn(self, fu, u, dealias=None):
        """ifft in three directions using mpi.
        Need to do ifft in reversed order of fft

        dealias = "3/2-rule"
            - Padded transform with 3/2-rule. fu is padded with zeros
              before transforming to complex space of shape original_shape_padded()
            - u is of original_shape_padded()
        
        dealias = "2/3-rule"
            - Transform is using 2/3-rule, i.e., frequencies higher than
              2/3*N are set to zero before transforming
            - u is of original_shape()
              
        dealias = None
            - Regular transform
            - u is of original_shape()
            
        fu is of transformed_shape()
        
        """
        assert dealias in ('3/2-rule', '2/3-rule', 'None', None)

        if dealias == '2/3-rule' and self.dealias.shape == (0,):
            self.dealias = self.get_dealias_filter()

        if self.num_processes == 1:
            if not dealias == '3/2-rule':
                if dealias == '2/3-rule':
                    fu *= self.dealias
                
                u = ifftn(fu, u, axes=(0,1,2), threads=self.threads)
            
            else:
                assert u.shape == self.original_shape_padded()

                # First create padded complex array and then perform irfftn
                fu_padded = self.work_arrays[(u, 0, False)]
                fu_padded[:self.N[0]/2, :self.N[1]/2, self.ks] = fu[:self.N[0]/2, :self.N[1]/2]
                fu_padded[:self.N[0]/2, -self.N[1]/2:, self.ks] = fu[:self.N[0]/2, self.N[1]/2:]
                fu_padded[-self.N[0]/2:, :self.N[1]/2, self.ks] = fu[self.N[0]/2:, :self.N[1]/2]
                fu_padded[-self.N[0]/2:, -self.N[1]/2:, self.ks] = fu[self.N[0]/2:, self.N[1]/2:]                                
                u[:] = ifftn(fu_padded*self.padsize**3, overwrite_input=True, axes=(0,1,2), threads=self.threads)
                
            return u
        
        if not dealias == '3/2-rule':
            if dealias == '2/3-rule':
                fu *= self.dealias
            
            # Intermediate work arrays required for transform
            Uc_hat  = self.work_arrays[(self.complex_shape(), self.complex, 0, False)]
            Uc_mpi  = self.work_arrays[((self.num_processes, self.Np[0], self.Np[1], self.Nf), self.complex, 0, False)]
            Uc_hatT = self.work_arrays[(self.complex_shape_T(), self.complex, 0, False)]

            # Do first owned direction
            Uc_hat = ifft(fu, Uc_hat, axis=0, threads=self.threads)
                
            if self.communication == 'alltoall':
                # Communicate all values
                self.comm.Alltoall([Uc_hat, self.mpitype], [Uc_mpi, self.mpitype])
                Uc_hatT[:] = np.rollaxis(Uc_mpi, 1).reshape(Uc_hatT.shape)
            
            else:
                Uc_send = Uc_hat.reshape((self.num_processes, self.Np[0], self.Np[1], self.Nf))
                for i in xrange(self.num_processes):
                    if not i == self.rank:
                        self.comm.Sendrecv_replace([Uc_send[i], self.mpitype], i, 0, i, 0)   
                    Uc_hatT[:, i*self.Np[1]:(i+1)*self.Np[1]] = Uc_send[i]
                
            # Do last two directions
            u[:] = ifft2(Uc_hatT, overwrite_input=True, axes=(1,2), threads=self.threads)

        else:
            # Intermediate work arrays required for transform
            Upad_hat  = self.work_arrays[(self.complex_shape_padded_0(), self.complex, 0, False)]
            U_mpi     = self.work_arrays[(self.complex_shape_padded_0_I(), self.complex, 0, False)]
            Upad_hat1 = self.work_arrays[(self.complex_shape_padded_1(), self.complex, 0, False)]
            Upad_hat2 = self.work_arrays[(self.complex_shape_padded_2(), self.complex, 0, False)]
            Upad_hat3 = self.work_arrays[(self.complex_shape_padded_3(), self.complex, 0, False)]

            # Expand in x-direction and perform ifft
            Upad_hat = self.copy_to_padded_x(fu, Upad_hat)
            Upad_hat[:] = ifft(Upad_hat*self.padsize, overwrite_input=True, axis=0)  
            
            # Communicate to distribute first dimension (like Fig. 2b but padded in x-dir and z-direction of full size)            
            self.comm.Alltoall([Upad_hat, self.mpitype], [U_mpi, self.mpitype])
            
            # Transpose data and pad in y-direction before doing ifft. Now data is padded in x and y 
            Upad_hat1[:] = np.rollaxis(U_mpi, 1).reshape(Upad_hat1.shape)
            Upad_hat2 = self.copy_to_padded_y(Upad_hat1, Upad_hat2)
            Upad_hat2[:] = ifft(Upad_hat2*self.padsize, overwrite_input=True, axis=1)
            
            # pad in z-direction and perform final ifft
            Upad_hat3 = self.copy_to_padded_z(Upad_hat2, Upad_hat3)
            u[:] = ifft(Upad_hat3*self.padsize, overwrite_input=True, axis=2, threads=self.threads)
            
        return u

    def fftn(self, u, fu, dealias=None):
        """fft in three directions using mpi
        
        dealias = "3/2-rule"
            - Truncated transform with 3/2-rule. The transfored fu is truncated
              when copied to complex space of complex_shape()
            - fu is of transformed_shape()
            - u is of original_shape_padded()
        
        dealias = "2/3-rule"
            - Regular transform
            - fu is of transformed_shape()
            - u is of original_shape()
              
        dealias = None
            - Regular transform
            - fu is of transformed_shape()
            - u is of original_shape()
            
        """
        assert dealias in ('3/2-rule', '2/3-rule', 'None', None)
        
        if self.num_processes == 1:
            if not dealias == '3/2-rule':
                assert u.shape == self.original_shape()
                
                fu = fftn(u, fu, axes=(0,1,2), threads=self.threads)
            
            else:
                assert u.shape == self.original_shape_padded()
                
                fu_padded = self.work_arrays[(u, 0, False)]
                fu_padded[:] = fftn(u/self.padsize**3, overwrite_input=True, axes=(0,1,2), threads=self.threads)
                
                # Copy with truncation
                fu[:self.N[0]/2, :self.N[1]/2] = fu_padded[:self.N[0]/2, :self.N[1]/2, self.ks]
                fu[:self.N[0]/2, self.N[1]/2:] = fu_padded[:self.N[0]/2, -self.N[1]/2:, self.ks]
                fu[self.N[0]/2:, :self.N[1]/2] = fu_padded[-self.N[0]/2:, :self.N[1]/2, self.ks]
                fu[self.N[0]/2:, self.N[1]/2:] = fu_padded[-self.N[0]/2:, -self.N[1]/2:, self.ks]
                                                
            return fu
        
        if not dealias == '3/2-rule':
            if self.communication == 'alltoall':
                # Intermediate work arrays required for transform
                Uc_mpi  = self.work_arrays[((self.num_processes, self.Np[0], self.Np[1], self.Nf), self.complex, 0, False)]
                Uc_hatT = self.work_arrays[(self.complex_shape_T(), self.complex, 0, False)]

                # Do 2 ffts in y-z directions on owned data
                Uc_hatT = fft2(u, Uc_hatT, axes=(1,2), threads=self.threads)
                
                # Transform data to align with x-direction  
                Uc_mpi[:] = np.rollaxis(Uc_hatT.reshape(self.Np[0], self.num_processes, self.Np[1], self.Nf), 1)
                    
                # Communicate all values
                self.comm.Alltoall([Uc_mpi, self.mpitype], [fu, self.mpitype])  
            
            else:
                # Communicating intermediate result 
                ft = fu.transpose(1,0,2)
                ft = fft2(u, ft, axes=(1,2), threads=self.threads)
                fu_send = fu.reshape((self.num_processes, self.Np[1], self.Np[1], self.Nf))
                for i in xrange(self.num_processes):
                    if not i == self.rank:
                        self.comm.Sendrecv_replace([fu_send[i], self.mpitype], i, 0, i, 0)   
                fu_send[:] = fu_send.transpose(0,2,1,3)
                            
            # Do fft for last direction 
            fu[:] = fft(fu, axis=0, threads=self.threads)
        
        else:
            # Intermediate work arrays required for transform
            Upad_hat  = self.work_arrays[(self.complex_shape_padded_0(), self.complex, 0, False)]
            Upad_hat0 = self.work_arrays[(self.complex_shape_padded_0(), self.complex, 1, False)]
            Upad_hat1 = self.work_arrays[(self.complex_shape_padded_1(), self.complex, 0, False)]
            Upad_hat3 = self.work_arrays[(self.complex_shape_padded_3(), self.complex, 0, False)]
            U_mpi     = self.work_arrays[(self.complex_shape_padded_0_I(), self.complex, 0, False)]

            # Do ffts in y and z directions
            Upad_hat3[:] = fft2(u/self.padsize**2, overwrite_input=True, axes=(1,2), threads=self.threads)   
            
            # Copy with truncation 
            Upad_hat1 = self.copy_from_padded(Upad_hat3, Upad_hat1)
            
            # Transpose and commuincate data
            U_mpi[:] = np.rollaxis(Upad_hat1.reshape(self.complex_shape_padded_I()), 1)
            self.comm.Alltoall([U_mpi, self.mpitype], [Upad_hat0, self.mpitype])
            
            # Perform fft of data in x-direction
            Upad_hat[:] = fft(Upad_hat0/self.padsize, overwrite_input=True, axis=0, threads=self.threads)
            
            # Truncate to original complex shape
            fu[:self.N[0]/2] = Upad_hat[:self.N[0]/2]
            fu[self.N[0]/2:] = Upad_hat[-self.N[0]/2:]
        
        return fu
    
    def copy_to_padded_x(self, fu, fp):
        fp[:self.N[0]/2] = fu[:self.N[0]/2]
        fp[-(self.N[0]/2):] = fu[self.N[0]/2:]
        return fp

    def copy_to_padded_y(self, fu, fp):
        fp[:, :self.N[1]/2] = fu[:, :self.N[1]/2]
        fp[:, -(self.N[1]/2):] = fu[:, self.N[1]/2:]
        return fp
    
    def copy_to_padded_z(self, fu, fp):
        fp[:, :, :self.N[2]/2] = fu[:, :, :self.N[2]/2]
        fp[:, :, -self.N[2]/2:] = fu[:, :, self.N[2]/2:]
        return fp
    
    def copy_from_padded(self, fp, fu):
        ks = (fftfreq(self.N[2])*self.N[2]).astype(int)        
        fu[:, :self.N[1]/2] = fp[:, :self.N[1]/2, ks]
        fu[:, self.N[1]/2:] = fp[:, -self.N[1]/2:, ks]
        return fu
