
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import pylab as pl
from scipy import signal as sg
from scipy import fftpack as spfft
import soundfile as sf

def calculate_perfect_sweet(M, factor,flag):
    p = []
    id = 1
    MM = M
    if flag == 0:
        p = []
        for M_i in range(1,M,2):
            if (M_i < 0):
                print("Please enter nyu value bigger than 0!")
                return 0;
            if(M_i < MM/2):
                m = MM / factor
                xx = np.exp(-1j*4*m*np.pi*np.square(M_i)/np.square(MM))
                p.append(xx)
                id = id + 2
            if(M_i >= MM/2):
                xx = np.conjugate(p[int(id-MM/2)])*(MM-M_i)
                p.append(xx)
                id = id + 2
    if flag == 1:
        p = []
        id = 0
        MM = M
        for M_i in range(0,M,2):
            if (M_i < 0):
                print("Please enter nyu value bigger than 0!")
                return 0;
            if(M_i < MM/2):
                m = MM / factor
                xx = np.exp(-1j*4*m*np.pi*np.square(M_i)/np.square(MM))
                p.append(xx)
                id = id + 2
            if(M_i >= MM/2):
                xx = np.conjugate(p[int(id-MM/2)])*(MM-M_i)
                p.append(xx)
                id = id + 2
    return p

def DFT_time_Domain(M, factor, flag):
    p = calculate_perfect_sweet(M, factor, flag)
    fft_sig = spfft.fft(p, n=len(p))
    return fft_sig


def IDFT_freq_Domain(signal):
    ifft_sig = spfft.ifft(signal,n=len(signal))
    return ifft_sig





#M_array_odd = np.arange(1,308)
#M_array_even = np.arange(1,309)

M_odd = 309
M_even = 308

#M_odd_np = np.array(M_array_odd, np.dtype(int))
#M_even_np = np.array(M_array_even, np.dtype(int))
#m_val = M/2

#############odd_M###########################
factor = 2
flag = 0##odd number
freq_domain_signal_odd = DFT_time_Domain(M_odd, factor, flag)
time_domain_signal_odd = IDFT_freq_Domain(freq_domain_signal_odd)
#############################################
#############even_M###########################
factor = 2
flag = 1#even number
freq_domain_signal_even = DFT_time_Domain(M_even, factor, flag)
time_domain_signal_even = IDFT_freq_Domain(freq_domain_signal_even)

#############################################




