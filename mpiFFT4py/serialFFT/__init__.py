try:
    assert False   # There is some issue with padding and pyfftw
    from pyfftw_fft import *
   
except:
    from numpy_fft import *
