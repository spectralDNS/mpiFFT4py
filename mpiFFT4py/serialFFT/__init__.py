import warnings
try:
    from pyfftw_fft import *
except:
    warnings.warn("Using numpy for FFTs, this is probably slower than using pyfftw and multithreading is not supported")
    from numpy_fft import *
