# -*- coding: utf-8 -*-

import pylab as pl
from scipy import interpolate
import matplotlib.pyplot as plt

from scipy.interpolate import Rbf, InterpolatedUnivariateSpline

import numpy as np
import scipy.signal as sig


# Perfect Sweep
def perfect_sweep(N):
    """
    generate_PerfectSweep returns a periodic perfect sweep

     Parametrs
     ---------
    N :     int
            length of the perfect sequence / sample

     Returns

     p :     array
             perfect_sweep

    """

    m = np.arange(0, np.ceil(N / 2 + 1))
    P_half = np.exp(-1j * 2 * np.pi / N * m ** 2)
    return np.real(np.fft.irfft(P_half, n=N))


def perfect_sequence_randomphase(N):
    """
     Parametrs
     ---------
     N :     int
             length of the perfect sequence / sample

     Returns

     p :     array
             perfect_sweep

   """

    m = np.arange(0, np.ceil(N / 2 + 1))
    phase = 2 * np.pi * np.random.random(len(m))
    phase[0] = 0
    P_half = np.exp(-1j * phase)
    if (N % 2) == 0:
        P_half[-1] = 1

    return np.fft.irfft(P_half, n=N)


def cconv(x, y, N=None):
    return np.fft.irfft(np.fft.rfft(x, n=N) * np.fft.rfft(y, n=N), n=N)


def cxcorr(x, y, N=None):
    return np.fft.irfft(np.fft.rfft(x) * np.fft.rfft(np.roll(y[::-1], 1)))


def time_reverse(x):
    N = len(x)
    return np.roll(x, -1)[N - 1::-1]


def db(x):
    return 20 * np.log10(np.abs(x))


def lagr_poly(ni, n):
    """Lagrange polynomail of order n

     Parameters
     ----------
    ni : array
          Sequences
     n  : scalar
          input

     Returns
     -------
     h : array
         Lagrange polynomial

     Notes
     -----

    """
    N = len(ni)
    h = np.zeros(N)
    for m in range(N):
        nm = ni[m]
        idx = np.concatenate([np.arange(0, m), np.arange(m + 1, N)])
        h[m] = np.prod((n - ni[idx]) / (nm - ni[idx]))
    return h


def fdfilt_lagr(tau, Lf, fs):
    """
     Parameters
     ----------
     tau : delay / s
     Lf : length of the filter / sample
     fs : sampling rate / Hz

     Returns
     -------
     h : (Lf)
         nonzero filter coefficients
     ni : time index of the first element of h
     n0 : time index of the center of h

    """

    d = tau * fs

    if Lf % 2 == 0:
        n0 = np.ceil(d)
        Lh = int(Lf / 2)
        idx = np.arange(n0 - Lh, n0 + Lh).astype(int)
    elif Lf % 2 == 1:
        n0 = np.round(d)
        Lh = int(np.floor(Lf / 2))
        idx = np.arange(n0 - Lh, n0 + Lh + 1).astype(int)
    else:
        print('Invalid value of Lf. Must be an integer')
    return lagr_poly(idx, d), idx[0], n0


def fdfilt_sinc(tau, Lf, fs, beta=8.6):
    """
     Parameters
     ----------
     tau : delay / s
     Lf : length of the filter / sample
     fs : sampling rate / Hz

     Returns
     -------
     h : (Lf)
         nonzero filter coefficients
     ni : time index of the first element of h
     n0 : time index of the center of h

    """

    d = tau * fs
    w = np.kaiser(Lf, beta)

    if Lf % 2 == 0:
        n0 = np.ceil(d)
        Lh = int(Lf / 2)
        idx = np.arange(n0 - Lh, n0 + Lh).astype(int)
    elif Lf % 2 == 1:
        n0 = np.round(d)
        Lh = int(np.floor(Lf / 2))
        idx = np.arange(n0 - Lh, n0 + Lh + 1).astype(int)
    else:
        print('Invalid value of Lf. Must be an integer')

    return np.sinc(idx - d) * w, idx[0], n0


def fdfilter(xi, yi, x, order, type='lagrange'):
    """
     Lagrange interpolation

     Parameters
     ----------
     xi :
         in accending order
     yi :

     x  :
         [xmin, xmax]

     Return
     ------
     yi :

    """
    N = order + 1
    if N % 2 == 0:
        Nhalf = N / 2
        n0 = np.searchsorted(xi, x)
        idx = np.linspace(n0 - Nhalf, n0 + Nhalf, num=N, endpoint=False).astype(int)
    elif N % 2 == 1:
        Nhalf = (N - 1) / 2
        n0 = np.argmin(np.abs(xi - x))
        idx = np.linspace(n0 - Nhalf, n0 + Nhalf + 1, num=N, endpoint=False).astype(int)
    else:
        print('order must be an integer')

    return np.dot(yi[idx], lagr_poly(xi[idx], x))


def fractional_delay(delay, Lf, fs, type):
    """
    fractional delay filter

    Parameters
    ----------
    delay : array
            time-varying delay in sample
    Lf    : int
            length of the fractional delay filter

    Returns
    -------
    waveform : array (Lf)
                nonzero coefficients
    shift    : array (Lf)
                indices of the first nonzero coefficient
    offset   : array (Lf)
                indices of the center of the filter
    """
    L = len(delay)
    waveform = np.zeros((L, Lf))
    shift = np.zeros(L)
    offset = np.zeros(L)

    if type == 'sinc':
        for n in range(L):
            htemp, ni, n0 = fdfilt_sinc(delay[n], Lf, fs=fs)
            waveform[n, :] = htemp
            shift[n] = ni
            offset[n] = n0
    elif type == 'lagrange':
        for n in range(L):
            htemp, ni, n0 = fdfilt_lagr(delay[n], Lf, fs=fs)
            waveform[n, :] = htemp
            shift[n] = ni
            offset[n] = n0
    else:
        print('unknown type')
    return waveform, shift, offset


def construct_ir_matrix(waveform, shift, Nh):
    """
    Convert 'waveform' and 'shift' into an IR matrix

    Parameters
    ----------
    waveform : array
                nonzero elements of the IRs
    shift :    array
                indices of the first nonzero coefficients
    Nh :       int
               length of each IRs

    Returns
    -------
    h :
       IRs
    H :
         TFs
    Ho :
         CHT spectrum


    """
    L, Lf = waveform.shape
    h = np.zeros((L, Nh))
    for n in range(L):
        idx = (np.arange(shift[n], shift[n] + Lf)).astype(int)
        h[n, idx] = waveform[n, :]
    H = np.fft.fft(h)
    Ho = (1 / L) * np.roll(np.fft.fft(H, axis=0), int(L / 2), axis=0)
    return h, H, Ho


def captured_signal(waveform, shift, p):
    """
    Apply time-varying delay to a perfect sweep

    Parameters
    ----------
    waveform : array
                nonzero filter coefficients
    shift :    array
                indices of the first nonzero coefficients
    p :        array
                periodic excitation signal

    Returns
    -------
    s : array
                captured signal

    """
    return time_varying_delay(waveform, shift, p)


def time_varying_delay(waveform, shift, p):
    """
    Apply a time varying delay to an input sequence
    """
    L, Lf = waveform.shape
    N = len(p)
    s = np.zeros(L)
    for n in range(L):
        idx = np.arange(shift[n], shift[n] + Lf).astype(int)
        s[n] = np.dot(p[np.mod(n - idx, N)], waveform[n, :])
    return s  # Constants


def nearestneighbour(ni_i, phi_i):
    """

    """
    h = interpolate.NearestNDInterpolator(ni_i, phi_i)
    return h.x


def spatial_interpolation(s_i, phi_i, phi_target, interp_method):
    s_i = np.array(s_i)
    phi_i = np.array(phi_i)
    phi_target = np.array(phi_target)

    if interp_method == 'linear':
        f = interpolate.interp1d(phi_i, s_i, bounds_error=False)
        h = f(phi_target)
    elif interp_method == 'spline':
        tck = interpolate.interp1d(phi_i, s_i, kind = 'cubic',bounds_error=False)
        h = tck(phi_target)
    elif interp_method == 'fitpack2 method':
        ius = InterpolatedUnivariateSpline(phi_i, s_i)
        h = ius(phi_target)
    else:
        print("Please select correct interpolation method")
        return
    return h




# Define a microphone movement (radius, angular speed, etc)
# The angular position of the microphone phi (length of L)
# Gerenate a perfect sequence p (length N)
# Compute the captured signal s, which has the same length L


# Constants
c = 343

# Parameters
N = 800
Nh = int(N / 2)
Q = 12
Omega = 2 * np.pi / Q
K = N
Lf = 13
fs = 8000
L = 2 * np.pi / Omega * fs
time = np.arange(np.ceil(L)) / fs

R = 0.5

# Source: You can change the number of microphones
xs = [0, 2]  # Point source

L = int(2 * np.pi / Omega * fs)
t = (1 / fs) * np.arange(L)
phi = Omega * t

#Define mic position
xm = [R * np.cos(phi), R * np.sin(phi)]

rm = np.sqrt((xm[0] - xs[0]) ** 2 + (xm[1] - xs[1]) ** 2)
delay = rm / c

# Excitation by perfet sequences.
#p = perfect_sequence_randomphase(N)
p = perfect_sweep(N)

type = 'lagrange'  # FD filters

######################System Identification######################




# initializing of spatial interplation values
y_i = np.zeros(N)

# getting impulse_respones
waveform, shift, offset = fractional_delay(delay, Lf, fs=fs, type=type)
# getting captured signal for each microphone
s = captured_signal(waveform, shift, p)

#################Select interpolation method####################################
interp_method = 'spline'
#interp_method = 'linear'
#interp_method = 'fitpack2 method'


################################################################################

#setting target phases
phi_target = np.linspace(0, 2*np.pi, num=K)

#number of subsignals
M = 4

#initializing of impulse_response
impulse_response = np.zeros((M,K-2))

#for each subsignal
for i in range(M):

    #Decompose the captured signal into M sub-signals
    phi_i = phi[i::M]
    s_i = s[i::M]

    #interpolation
    y_i = spatial_interpolation(s_i, phi_i, phi_target, interp_method)


    #print(y_i)
    y_i = y_i[1:len(y_i)-1] #removing of boundary valuse, bc these are nan.
    p_i = p[1:len(p) - 1]
    phi_target_i = phi_target[1:len(phi_target)-1]

    #calculating of impulse_response
    impulse_response[i,:] = cross_correlation(y_i, p_i)

    plt.close()
    plt.plot(phi_target_i, impulse_response[i,:],'r',phi_target_i, y_i,'b')
    plt.xlabel('phi')
    plt.ylabel('Impulse Responses')
    plt.title('Impulse Responses')
    plt.grid = True
    plt.legend(['impulse_response', 'cubic interpolations'], loc='best')
    plt.show()














    ##################################################################
