from __future__ import division
"""Slab decomposition

This module contains classes for performing FFTs with slab decomposition
of three-dimensional data structures data[Nx, Ny, Nz], where (Nx, Ny, Nz) is
the shape of the input data. With slab decomposition only one of these three
indices is shared, leading to local datastructures on each processor
with shape data[Nx/P, Ny, Nz], where P is the total number of processors.

classes:
    R2C - For real to complex transforms
    C2C - For complex to complex transforms
"""
__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2016-02-16"
__copyright__ = "Copyright (C) 2016 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"

from .serialFFT import *
import numpy as np
from .mpibase import work_arrays, datatypes
from numpy.fft import fftfreq, rfftfreq
from .cython.maths import dealias_filter, transpose_Uc #, transpose_Umpi
from collections import defaultdict
from mpi4py import MPI

# Using Lisandro Dalcin's code for Alltoallw.
# Note that _subsize and _distribution are only really required for
# general shape meshes. Here we require power two.

def _subsize(N, size, rank):
    return N // size + (N % size > rank)

def _distribution(N, size):
    q = N // size
    r = N % size
    n = s = i = 0
    while i < size:
        n = q
        s = q * i
        if i < r:
            n += 1
            s += i
        else:
            s += r
        yield n, s
        i += 1

class R2C(object):
    """Class for performing FFT in 3D using MPI

    Slab decomposition

    Args:
        N - NumPy array([Nx, Ny, Nz]) Number of nodes for the real mesh
        L - NumPy array([Lx, Ly, Lz]) The actual size of the real mesh
        comm - The MPI communicator object
        precision - "single" or "double"
        communication - Method used for communication ('Alltoall', 'Sendrecv_replace', 'Alltoallw')
        padsize - Padsize when dealias = 3/2-rule is used
        threads - Number of threads used by FFTs
        planner_effort - Planner effort used by FFTs (e.g., "FFTW_MEASURE", "FFTW_PATIENT", "FFTW_EXHAUSTIVE")
                         Give as defaultdict, with keys representing transform (e.g., fft, ifft)

    The forward transform is real to complex and the inverse is complex to real
    """
    def __init__(self, N, L, comm, precision,
                 communication="Alltoallw",
                 padsize=1.5,
                 threads=1,
                 planner_effort=defaultdict(lambda: "FFTW_MEASURE")):
        assert len(L) == 3
        assert len(N) == 3
        self.N = N
        self.Nf = N[2]//2+1          # Independent complex wavenumbers in z-direction
        self.Nfp = int(padsize*N[2]//2+1) # Independent complex wavenumbers in z-direction for padded array
        self.comm = comm
        self.float, self.complex, self.mpitype = datatypes(precision)
        self.communication = communication
        self.num_processes = comm.Get_size()
        self.rank = comm.Get_rank()
        self.Np = N // self.num_processes
        self.L = L.astype(self.float)
        self.dealias = np.zeros(0)
        self.padsize = padsize
        self.threads = threads
        self.planner_effort = planner_effort
        self.work_arrays = work_arrays()
        if not self.num_processes in [2**i for i in range(int(np.log2(N[0]))+1)]:
            raise IOError("Number of cpus must be in ",
                          [2**i for i in range(int(np.log2(N[0]))+1)])
        self._subarraysA = []
        self._subarraysB = []
        self._counts_displs = 0
        self._subarraysA_pad = []
        self._subarraysB_pad = []

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

    def global_complex_shape(self, padsize=1.):
        """Global size of problem in complex wavenumber space"""
        return (int(padsize*self.N[0]), int(padsize*self.N[1]),
                int(padsize*self.N[2]//2+1))

    def work_shape(self, dealias):
        """Shape of work arrays used in convection with dealiasing.

        Note the different shape whether or not padding is involved.
        """
        if dealias == '3/2-rule':
            return self.real_shape_padded()

        else:
            return self.real_shape()

    def real_local_slice(self, padsize=1):
        """Local slice in real space of the input array

        Array can be padded with padsize > 1
        """
        return (slice(int(padsize*self.rank*self.Np[0]),
                      int(padsize*(self.rank+1)*self.Np[0]), 1),
                slice(0, int(padsize*self.N[1]), 1),
                slice(0, int(padsize*self.N[2]), 1))

    def complex_local_slice(self):
        """Local slice of complex return array"""
        return (slice(0, self.N[0], 1),
                slice(self.rank*self.Np[1], (self.rank+1)*self.Np[1], 1),
                slice(0, self.Nf, 1))

    def complex_local_wavenumbers(self):
        """Returns local wavenumbers of complex space"""
        return (fftfreq(self.N[0], 1./self.N[0]),
                fftfreq(self.N[1], 1./self.N[1])[self.complex_local_slice()[1]],
                rfftfreq(self.N[2], 1./self.N[2]))

    def get_local_mesh(self):
        """Returns the local decomposed physical mesh"""
        X = np.ogrid[self.rank*self.Np[0]:(self.rank+1)*self.Np[0],
                     :self.N[1], :self.N[2]]
        X[0] = (X[0]*self.L[0]/self.N[0]).astype(self.float)
        X[1] = (X[1]*self.L[1]/self.N[1]).astype(self.float)
        X[2] = (X[2]*self.L[2]/self.N[2]).astype(self.float)
        X = [np.broadcast_to(x, self.real_shape()) for x in X]
        return X

    def get_local_wavenumbermesh(self, scaled=False):
        """Returns (scaled) local decomposed wavenumbermesh

        If scaled is True, then the wavenumbermesh is scaled with physical mesh
        size. This takes care of mapping the physical domain to a computational
        cube of size (2pi)**3
        """
        kx, ky, kz = self.complex_local_wavenumbers()
        Ks = np.meshgrid(kx, ky, kz, indexing='ij', sparse=True)
        if scaled:
            Lp = 2*np.pi/self.L
            for i in range(3):
                Ks[i] *= Lp[i]
        K = [np.broadcast_to(k, self.complex_shape()) for k in Ks]
        #K = np.array(np.meshgrid(kx, ky, kz, indexing='ij')).astype(self.float)
        return K

    def get_dealias_filter(self):
        """Filter for dealiasing nonlinear convection"""
        K = self.get_local_wavenumbermesh()
        kmax = 2./3.*(self.N//2+1)
        dealias = np.array((abs(K[0]) < kmax[0])*(abs(K[1]) < kmax[1])*
                           (abs(K[2]) < kmax[2]), dtype=np.uint8)
        return dealias

    def get_subarrays(self, padsize=1):
        """Subarrays for Alltoallw transforms"""
        datatype = MPI._typedict[np.dtype(self.complex).char]
        _subarraysA = [
            datatype.Create_subarray([int(padsize*self.N[0]), self.Np[1], self.Nf], [l, self.Np[1], self.Nf], [s, 0, 0]).Commit()
            for l, s in _distribution(int(padsize*self.N[0]), self.num_processes)
        ]
        _subarraysB = [
            datatype.Create_subarray([int(padsize*self.Np[0]), self.N[1], self.Nf], [int(padsize*self.Np[0]), l, self.Nf], [0, s, 0]).Commit()
            for l, s in _distribution(self.N[1], self.num_processes)
        ]
        _counts_displs = ([1] * self.num_processes, [0] * self.num_processes)
        return _subarraysA, _subarraysB, _counts_displs

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
                u = irfftn(fu, u, axes=(0, 1, 2), threads=self.threads, planner_effort=self.planner_effort['irfftn'])

            else:
                assert u.shape == self.real_shape_padded()

                # Scale smallest array with padsize
                fu_ = self.work_arrays[(fu, 0, False)]
                fu_[:] = fu*self.padsize**3

                # First create padded complex array and then perform irfftn
                fu_padded = self.work_arrays[(self.global_complex_shape(padsize=1.5), self.complex, 0)]
                fu_padded[:self.N[0]//2, :self.N[1]//2, :self.Nf] = fu_[:self.N[0]//2, :self.N[1]//2]
                fu_padded[:self.N[0]//2, -self.N[1]//2:, :self.Nf] = fu_[:self.N[0]//2, self.N[1]//2:]
                fu_padded[-self.N[0]//2:, :self.N[1]//2, :self.Nf] = fu_[self.N[0]//2:, :self.N[1]//2]
                fu_padded[-self.N[0]//2:, -self.N[1]//2:, :self.Nf] = fu_[self.N[0]//2:, -self.N[1]//2:]

                ## Current transform is only exactly reversible if periodic transforms are made symmetric
                ## However, this seems to lead to more aliasing and as such the non-symmetrical padding is used
                #fu_padded[:, -self.N[1]//2] *= 0.5
                #fu_padded[-self.N[0]//2] *= 0.5
                #fu_padded[self.N[0]//2] = fu_padded[-self.N[0]//2]
                #fu_padded[:, self.N[1]//2] = fu_padded[:, -self.N[1]//2]

                u[:] = irfftn(fu_padded, overwrite_input=True,
                              axes=(0, 1, 2), threads=self.threads,
                              planner_effort=self.planner_effort['irfftn'])
            return u

        if not dealias == '3/2-rule':
            # Intermediate work arrays required for transform
            Uc_hat = self.work_arrays[(self.complex_shape(), self.complex, 0, False)]

            # Do first owned direction
            Uc_hat = ifft(fu, Uc_hat, axis=0, threads=self.threads, planner_effort=self.planner_effort['ifft'])

            if self.communication == 'Alltoall':
                Uc_mpi = self.work_arrays[((self.num_processes, self.Np[0], self.Np[1], self.Nf), self.complex, 0, False)]

                ## Communicate all values
                self.comm.Alltoall([Uc_hat, self.mpitype], [Uc_mpi, self.mpitype])
                #Uc_hatT = np.rollaxis(Uc_mpi, 1).reshape(self.complex_shape_T())
                Uc_hatT = self.work_arrays[(self.complex_shape_T(), self.complex, 0, False)]
                Uc_hatT = transpose_Uc(Uc_hatT, Uc_mpi, self.num_processes, self.Np[0], self.Np[1], self.Nf)

                #self.comm.Alltoall(MPI.IN_PLACE, [Uc_hat, self.mpitype])
                #Uc_hatT = np.rollaxis(Uc_hat.reshape((self.num_processes, self.Np[0], self.Np[1], self.Nf)), 1).reshape(self.complex_shape_T())

            elif self.communication == 'Sendrecv_replace':
                Uc_send = Uc_hat.reshape((self.num_processes, self.Np[0], self.Np[1], self.Nf))
                Uc_hatT = self.work_arrays[(self.complex_shape_T(), self.complex, 0, False)]
                for i in xrange(self.num_processes):
                    if not i == self.rank:
                        self.comm.Sendrecv_replace([Uc_send[i], self.mpitype], i, 0, i, 0)
                    Uc_hatT[:, i*self.Np[1]:(i+1)*self.Np[1]] = Uc_send[i]

            elif self.communication == 'Alltoallw':
                if len(self._subarraysA) == 0:
                    self._subarraysA, self._subarraysB, self._counts_displs = self.get_subarrays()
                Uc_hatT = self.work_arrays[(self.complex_shape_T(), self.complex, 0, False)]
                self.comm.Alltoallw(
                    [Uc_hat, self._counts_displs, self._subarraysA],
                    [Uc_hatT, self._counts_displs, self._subarraysB])

            # Do last two directions
            u = irfft2(Uc_hatT, u, overwrite_input=True, axes=(1, 2),
                       threads=self.threads,
                       planner_effort=self.planner_effort['irfft2'])

        else:
            assert self.num_processes <= self.N[0]//2, "Number of processors cannot be larger than N[0]//2 for 3/2-rule"

            # Intermediate work arrays required for transform
            Upad_hat  = self.work_arrays[(self.complex_shape_padded_0(), self.complex, 0)]
            Upad_hat1 = self.work_arrays[(self.complex_shape_padded_1(), self.complex, 0, False)]
            Upad_hat2 = self.work_arrays[(self.complex_shape_padded_2(), self.complex, 0)]
            Upad_hat3 = self.work_arrays[(self.complex_shape_padded_3(), self.complex, 0)]

            # Expand in x-direction and perform ifft
            Upad_hat = R2C.copy_to_padded(fu*self.padsize**3, Upad_hat, self.N, axis=0)
            Upad_hat[:] = ifft(Upad_hat, axis=0, threads=self.threads,
                               planner_effort=self.planner_effort['ifft'])

            if not self.communication == 'Alltoallw':
                # Communicate to distribute first dimension (like Fig. 2b but padded in x-dir)
                self.comm.Alltoall(MPI.IN_PLACE, [Upad_hat, self.mpitype])
                Upad_hat1[:] = np.rollaxis(Upad_hat.reshape(self.complex_shape_padded_0_I()), 1).reshape(Upad_hat1.shape)

            else:
                if len(self._subarraysA_pad) == 0:
                    self._subarraysA_pad, self._subarraysB_pad, self._counts_displs = self.get_subarrays(padsize=self.padsize)
                self.comm.Alltoallw(
                    [Upad_hat, self._counts_displs, self._subarraysA_pad],
                    [Upad_hat1, self._counts_displs, self._subarraysB_pad])

            # Transpose data and pad in y-direction before doing ifft. Now data is padded in x and y
            Upad_hat2 = R2C.copy_to_padded(Upad_hat1, Upad_hat2, self.N, axis=1)
            Upad_hat2[:] = ifft(Upad_hat2, axis=1, threads=self.threads,
                                planner_effort=self.planner_effort['ifft'])

            # pad in z-direction and perform final irfft
            Upad_hat3 = R2C.copy_to_padded(Upad_hat2, Upad_hat3, self.N, axis=2)
            u[:] = irfft(Upad_hat3, overwrite_input=True, axis=2, threads=self.threads,
                         planner_effort=self.planner_effort['irfft'])

        return u

    #@profile
    def fftn(self, u, fu, dealias=None):
        """fft in three directions using mpi

        dealias = "3/2-rule"
            - Truncated transform with 3/2-rule. The transformed fu is truncated
              when copied to complex space of complex_shape()
            - fu is of complex_shape()
            - u is of real_shape_padded()

        dealias = "2/3-rule" or None
            - Regular transform
            - fu is of complex_shape()
            - u is of real_shape()

        """
        assert dealias in ('3/2-rule', '2/3-rule', 'None', None)

        if self.num_processes == 1:
            if not dealias == '3/2-rule':
                assert u.shape == self.real_shape()
                fu = rfftn(u, fu, axes=(0, 1, 2), threads=self.threads,
                           planner_effort=self.planner_effort['rfftn'])

            else:
                assert u.shape == self.real_shape_padded()

                fu_padded = self.work_arrays[(self.global_complex_shape(padsize=1.5),
                                              self.complex, 0, False)]
                fu_padded = rfftn(u, fu_padded, axes=(0, 1, 2),
                                  planner_effort=self.planner_effort['rfftn'])

                # Copy with truncation
                fu[:self.N[0]//2, :self.N[1]//2] = fu_padded[:self.N[0]//2, :self.N[1]//2, :self.Nf]
                fu[:self.N[0]//2, self.N[1]//2:] = fu_padded[:self.N[0]//2, -self.N[1]//2:, :self.Nf]
                fu[self.N[0]//2:, :self.N[1]//2] = fu_padded[-self.N[0]//2:, :self.N[1]//2, :self.Nf]
                fu[self.N[0]//2:, self.N[1]//2:] = fu_padded[-self.N[0]//2:, -self.N[1]//2:, :self.Nf]
                fu /= self.padsize**3
                ## Modify for symmetric padding
                #fu[:, -self.N[1]//2] *= 2
                #fu[self.N[0]//2] *= 2

            return fu

        if not dealias == '3/2-rule':

            Uc_hat = self.work_arrays[(fu, 0, False)]

            if self.communication == 'Alltoall':
                # Intermediate work arrays required for transform
                Uc_hatT = self.work_arrays[(self.complex_shape_T(), self.complex, 0, False)]
                U_mpi = self.work_arrays[((self.num_processes, self.Np[0], self.Np[1], self.Nf), self.complex, 0, False)]

                # Do 2 ffts in y-z directions on owned data
                Uc_hatT = rfft2(u, Uc_hatT, axes=(1, 2), threads=self.threads, planner_effort=self.planner_effort['rfft2'])

                #Transform data to align with x-direction
                U_mpi[:] = np.rollaxis(Uc_hatT.reshape(self.Np[0], self.num_processes, self.Np[1], self.Nf), 1)

                #Communicate all values
                self.comm.Alltoall([U_mpi, self.mpitype], [Uc_hat, self.mpitype])

                ## Transform data to align with x-direction
                #U_mpi = transpose_Umpi(U_mpi, Uc_hatT, self.num_processes, self.Np[0], self.Np[1], self.Nf)

                ## Communicate all values
                #self.comm.Alltoall([U_mpi, self.mpitype], [fu, self.mpitype])

            elif self.communication == 'Sendrecv_replace':
                # Communicating intermediate result
                ft = Uc_hat.transpose(1, 0, 2)
                ft = rfft2(u, ft, axes=(1, 2), threads=self.threads,
                           planner_effort=self.planner_effort['rfft2'])
                fu_send = Uc_hat.reshape((self.num_processes, self.Np[1],
                                          self.Np[1], self.Nf))
                for i in xrange(self.num_processes):
                    if not i == self.rank:
                        self.comm.Sendrecv_replace([fu_send[i], self.mpitype], i, 0, i, 0)
                fu_send[:] = fu_send.transpose(0, 2, 1, 3)

            elif self.communication == 'Alltoallw':
                if len(self._subarraysA) == 0:
                    self._subarraysA, self._subarraysB, self._counts_displs = self.get_subarrays()

                # Intermediate work arrays required for transform
                Uc_hatT = self.work_arrays[(self.complex_shape_T(), self.complex, 0, False)]

                # Do 2 ffts in y-z directions on owned data
                Uc_hatT = rfft2(u, Uc_hatT, axes=(1, 2), threads=self.threads,
                                planner_effort=self.planner_effort['rfft2'])

                self.comm.Alltoallw(
                    [Uc_hatT, self._counts_displs, self._subarraysB],
                    [Uc_hat, self._counts_displs, self._subarraysA])

            # Do fft for last direction
            fu = fft(Uc_hat, fu, overwrite_input=True, axis=0,
                     threads=self.threads, planner_effort=self.planner_effort['fft'])

        else:
            assert self.num_processes <= self.N[0]//2, "Number of processors cannot be larger than N[0]//2 for 3/2-rule"
            assert u.shape == self.real_shape_padded()

            # Intermediate work arrays required for transform
            Upad_hat  = self.work_arrays[(self.complex_shape_padded_0(), self.complex, 0, False)]
            Upad_hat0 = self.work_arrays[(self.complex_shape_padded_0(), self.complex, 1, False)]
            Upad_hat1 = self.work_arrays[(self.complex_shape_padded_1(), self.complex, 0, False)]
            Upad_hat3 = self.work_arrays[(self.complex_shape_padded_3(), self.complex, 0, False)]

            # Do ffts in the padded y and z directions
            Upad_hat3 = rfft2(u, Upad_hat3, axes=(1, 2), threads=self.threads,
                              planner_effort=self.planner_effort['rfft2'])

            # Copy with truncation
            Upad_hat1 = R2C.copy_from_padded(Upad_hat3, Upad_hat1, self.N, 1)

            if self.communication == 'Alltoall':
                # Transpose and commuincate data
                Upad_hat0[:] = np.rollaxis(Upad_hat1.reshape(self.complex_shape_padded_I()), 1).reshape(Upad_hat0.shape)
                self.comm.Alltoall(MPI.IN_PLACE, [Upad_hat0, self.mpitype])

            elif self.communication == 'Alltoallw':
                if len(self._subarraysA_pad) == 0:
                    self._subarraysA_pad, self._subarraysB_pad, self._counts_displs = self.get_subarrays(padsize=self.padsize)

                self.comm.Alltoallw(
                    [Upad_hat1, self._counts_displs, self._subarraysB_pad],
                    [Upad_hat0, self._counts_displs, self._subarraysA_pad])

            # Perform fft of data in x-direction
            Upad_hat = fft(Upad_hat0, Upad_hat, axis=0, threads=self.threads,
                           planner_effort=self.planner_effort['fft'])

            # Truncate to original complex shape
            fu[:self.N[0]//2] = Upad_hat[:self.N[0]//2]
            fu[self.N[0]//2:] = Upad_hat[-self.N[0]//2:]
            fu /= self.padsize**3

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

    @staticmethod
    def copy_to_padded(fu, fp, N, axis=0):
        if axis == 0:
            fp[:N[0]//2] = fu[:N[0]//2]
            fp[-N[0]//2:] = fu[N[0]//2:]
        elif axis == 1:
            fp[:, :N[1]//2] = fu[:, :N[1]//2]
            fp[:, -N[1]//2:] = fu[:, N[1]//2:]
        elif axis == 2:
            fp[:, :, :(N[2]//2+1)] = fu[:]
        return fp

    @staticmethod
    def copy_from_padded(fp, fu, N, axis=0):
        if axis == 1:
            fu[:, :N[1]//2] = fp[:, :N[1]//2, :(N[2]//2+1)]
            fu[:, N[1]//2:] = fp[:, -N[1]//2:, :(N[2]//2+1)]
        elif axis == 2:
            fu[:] = fp[:, :, :(N[2]//2+1)]
        return fu

class C2C(R2C):
    """Class for performing FFT in 3D using MPI

    Slab decomposition

    Args:
        N - NumPy array([Nx, Ny, Nz]) Number of nodes for the real mesh
        L - NumPy array([Lx, Ly, Lz]) The actual size of the real mesh
        comm - The MPI communicator object
        precision - "single" or "double"
        communication - Method used for communication ('Alltoall', 'Sendrecv_replace')
        padsize - Padsize when dealias = 3/2-rule is used
        threads - Number of threads used by FFTs
        planner_effort - Planner effort used by FFTs (e.g., "FFTW_MEASURE", "FFTW_PATIENT", "FFTW_EXHAUSTIVE")
                         Give as defaultdict, with keys representing transform (e.g., fft, ifft)

    The transform is complex to complex
    """
    def __init__(self, N, L, comm, precision,
                 communication="Alltoall",
                 padsize=1.5,
                 threads=1,
                 planner_effort=defaultdict(lambda: "FFTW_MEASURE")):
        R2C.__init__(self, N, L, comm, precision,
                     communication=communication,
                     padsize=padsize, threads=threads,
                     planner_effort=planner_effort)
        # Reuse all shapes from r2c transform R2C simply by resizing the final complex z-dimension:
        self.Nf = N[2]
        self.Nfp = int(self.padsize*self.N[2]) # Independent complex wavenumbers in z-direction for padded array

        # Rename since there's no real space
        self.original_shape_padded = self.real_shape_padded
        self.original_shape = self.real_shape
        self.transformed_shape = self.complex_shape
        self.original_local_slice = self.real_local_slice
        self.transformed_local_slice = self.complex_local_slice
        self.ks = (fftfreq(N[2])*N[2]).astype(int)

    def global_shape(self, padsize=1.):
        """Global size of problem in transformed space"""
        return (int(padsize*self.N[0]), int(padsize*self.N[1]),
                int(padsize*self.N[2]))

    def transformed_local_wavenumbers(self):
        return (fftfreq(self.N[0], 1./self.N[0]),
                fftfreq(self.N[1], 1./self.N[1])[self.transformed_local_slice()[1]],
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

                u = ifftn(fu, u, axes=(0, 1, 2), threads=self.threads,
                          planner_effort=self.planner_effort['ifftn'])

            else:
                assert u.shape == self.original_shape_padded()

                # First create padded complex array and then perform irfftn
                fu_padded = self.work_arrays[(u, 0)]
                fu_padded[:self.N[0]//2, :self.N[1]//2, self.ks] = fu[:self.N[0]//2, :self.N[1]//2]
                fu_padded[:self.N[0]//2, -self.N[1]//2:, self.ks] = fu[:self.N[0]//2, self.N[1]//2:]
                fu_padded[-self.N[0]//2:, :self.N[1]//2, self.ks] = fu[self.N[0]//2:, :self.N[1]//2]
                fu_padded[-self.N[0]//2:, -self.N[1]//2:, self.ks] = fu[self.N[0]//2:, self.N[1]//2:]
                u = ifftn(fu_padded*self.padsize**3, u, overwrite_input=True,
                          axes=(0, 1, 2), threads=self.threads,
                          planner_effort=self.planner_effort['ifftn'])

            return u

        if not dealias == '3/2-rule':
            if dealias == '2/3-rule':
                fu *= self.dealias

            # Intermediate work arrays required for transform
            Uc_hat  = self.work_arrays[(self.complex_shape(), self.complex, 0, False)]
            Uc_mpi  = self.work_arrays[((self.num_processes, self.Np[0], self.Np[1], self.Nf), self.complex, 0, False)]
            Uc_hatT = self.work_arrays[(self.complex_shape_T(), self.complex, 0, False)]

            # Do first owned direction
            Uc_hat = ifft(fu, Uc_hat, axis=0, threads=self.threads,
                          planner_effort=self.planner_effort['ifft'])

            if self.communication == 'Alltoall':
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
            u = ifft2(Uc_hatT, u, overwrite_input=True, axes=(1, 2),
                      threads=self.threads,
                      planner_effort=self.planner_effort['ifft2'])

        else:
            # Intermediate work arrays required for transform
            Upad_hat  = self.work_arrays[(self.complex_shape_padded_0(), self.complex, 0, False)]
            U_mpi     = self.work_arrays[(self.complex_shape_padded_0_I(), self.complex, 0, False)]
            Upad_hat1 = self.work_arrays[(self.complex_shape_padded_1(), self.complex, 0, False)]
            Upad_hat2 = self.work_arrays[(self.complex_shape_padded_2(), self.complex, 0, False)]
            Upad_hat3 = self.work_arrays[(self.complex_shape_padded_3(), self.complex, 0, False)]

            # Expand in x-direction and perform ifft
            Upad_hat = C2C.copy_to_padded(fu*self.padsize**3, Upad_hat, self.N, axis=0)
            Upad_hat[:] = ifft(Upad_hat, axis=0, threads=self.threads,
                               planner_effort=self.planner_effort['ifft'])

            # Communicate to distribute first dimension (like Fig. 2b but padded in x-dir and z-direction of full size)
            self.comm.Alltoall([Upad_hat, self.mpitype], [U_mpi, self.mpitype])

            # Transpose data and pad in y-direction before doing ifft. Now data is padded in x and y
            Upad_hat1[:] = np.rollaxis(U_mpi, 1).reshape(Upad_hat1.shape)
            Upad_hat2 = C2C.copy_to_padded(Upad_hat1, Upad_hat2, self.N, axis=1)
            Upad_hat2[:] = ifft(Upad_hat2, axis=1, threads=self.threads,
                                planner_effort=self.planner_effort['ifft'])

            # pad in z-direction and perform final ifft
            Upad_hat3 = C2C.copy_to_padded(Upad_hat2, Upad_hat3, self.N, axis=2)
            u = ifft(Upad_hat3, u, overwrite_input=True, axis=2,
                     threads=self.threads, planner_effort=self.planner_effort['ifft'])

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

                fu = fftn(u, fu, axes=(0, 1, 2), threads=self.threads,
                          planner_effort=self.planner_effort['fftn'])

            else:
                assert u.shape == self.original_shape_padded()

                fu_padded = self.work_arrays[(u, 0)]
                fu_padded = fftn(u, fu_padded, axes=(0, 1, 2), threads=self.threads,
                                 planner_effort=self.planner_effort['fftn'])

                # Copy with truncation
                fu[:self.N[0]//2, :self.N[1]//2] = fu_padded[:self.N[0]//2, :self.N[1]//2, self.ks]
                fu[:self.N[0]//2, self.N[1]//2:] = fu_padded[:self.N[0]//2, -self.N[1]//2:, self.ks]
                fu[self.N[0]//2:, :self.N[1]//2] = fu_padded[-self.N[0]//2:, :self.N[1]//2, self.ks]
                fu[self.N[0]//2:, self.N[1]//2:] = fu_padded[-self.N[0]//2:, -self.N[1]//2:, self.ks]
                fu /= self.padsize**3
            return fu

        if not dealias == '3/2-rule':
            if self.communication == 'Alltoall':
                # Intermediate work arrays required for transform
                Uc_mpi  = self.work_arrays[((self.num_processes, self.Np[0], self.Np[1], self.Nf), self.complex, 0, False)]
                Uc_hatT = self.work_arrays[(self.complex_shape_T(), self.complex, 0, False)]

                # Do 2 ffts in y-z directions on owned data
                Uc_hatT = fft2(u, Uc_hatT, axes=(1,2), threads=self.threads, planner_effort=self.planner_effort['fft2'])

                # Transform data to align with x-direction
                Uc_mpi[:] = np.rollaxis(Uc_hatT.reshape(self.Np[0], self.num_processes, self.Np[1], self.Nf), 1)

                # Communicate all values
                self.comm.Alltoall([Uc_mpi, self.mpitype], [fu, self.mpitype])

            else:
                # Communicating intermediate result
                ft = fu.transpose(1, 0, 2)
                ft = fft2(u, ft, axes=(1, 2), threads=self.threads,
                          planner_effort=self.planner_effort['fft2'])
                fu_send = fu.reshape((self.num_processes, self.Np[1],
                                      self.Np[1], self.Nf))
                for i in xrange(self.num_processes):
                    if not i == self.rank:
                        self.comm.Sendrecv_replace([fu_send[i], self.mpitype], i, 0, i, 0)
                fu_send[:] = fu_send.transpose(0, 2, 1, 3)

            # Do fft for last direction
            fu[:] = fft(fu, axis=0, threads=self.threads,
                        planner_effort=self.planner_effort['fft'])

        else:
            # Intermediate work arrays required for transform
            Upad_hat  = self.work_arrays[(self.complex_shape_padded_0(), self.complex, 0, False)]
            Upad_hat0 = self.work_arrays[(self.complex_shape_padded_0(), self.complex, 1, False)]
            Upad_hat1 = self.work_arrays[(self.complex_shape_padded_1(), self.complex, 0, False)]
            Upad_hat3 = self.work_arrays[(self.complex_shape_padded_3(), self.complex, 0, False)]
            U_mpi     = self.work_arrays[(self.complex_shape_padded_0_I(), self.complex, 0, False)]

            # Do ffts in y and z directions
            Upad_hat3 = fft2(u, Upad_hat3, axes=(1, 2), threads=self.threads,
                             planner_effort=self.planner_effort['fft2'])

            # Copy with truncation
            Upad_hat1 = C2C.copy_from_padded(Upad_hat3, Upad_hat1, self.N, 1)

            # Transpose and commuincate data
            U_mpi[:] = np.rollaxis(Upad_hat1.reshape(self.complex_shape_padded_I()), 1)
            self.comm.Alltoall([U_mpi, self.mpitype], [Upad_hat0, self.mpitype])

            # Perform fft of data in x-direction
            Upad_hat = fft(Upad_hat0, Upad_hat, overwrite_input=True, axis=0, threads=self.threads, planner_effort=self.planner_effort['fft'])

            # Truncate to original complex shape
            fu[:self.N[0]//2] = Upad_hat[:self.N[0]//2]
            fu[self.N[0]//2:] = Upad_hat[-self.N[0]//2:]
            fu /= self.padsize**3

        return fu

    @staticmethod
    def copy_to_padded(fu, fp, N, axis=0):
        if axis == 0:
            fp[:N[0]//2] = fu[:N[0]//2]
            fp[-N[0]//2:] = fu[N[0]//2:]
        elif axis == 1:
            fp[:, :N[1]//2] = fu[:, :N[1]//2]
            fp[:, -N[1]//2:] = fu[:, N[1]//2:]
        elif axis == 2:
            fp[:, :, :N[2]//2] = fu[:, :, :N[2]//2]
            fp[:, :, -N[2]//2:] = fu[:, :, N[2]//2:]
        return fp

    @staticmethod
    def copy_from_padded(fp, fu, N, axis=0):
        if axis == 1:
            fu[:, :N[1]//2, :N[2]//2] = fp[:, :N[1]//2, :N[2]//2]
            fu[:, :N[1]//2, N[2]//2:] = fp[:, :N[1]//2, -N[2]//2:]
            fu[:, N[1]//2:, :N[2]//2] = fp[:, -N[1]//2:, :N[2]//2]
            fu[:, N[1]//2:, N[2]//2:] = fp[:, -N[1]//2:, -N[2]//2:]

        return fu


#def transpose_Uc(Uc_hatT, U_mpi, num_processes, Np0, Np1, Nf):
    #for i in xrange(num_processes):
        #Uc_hatT[:, i*Np1:(i+1)*Np1] = U_mpi[i]
    #return Uc_hatT

#def transpose_Umpi(U_mpi, Uc_hatT, num_processes, Np0, Np1, Nf):
    #for i in xrange(num_processes):
        #U_mpi[i] = Uc_hatT[:, i*Np1:(i+1)*Np1]
    #return U_mpi
