"""
Created on Mon Nov  6 22:07:20 2017

@author: davidkumar
"""



import pylab as pl
from scipy import interpolate
import matplotlib.pyplot as plt

from scipy.interpolate import Rbf, InterpolatedUnivariateSpline

import numpy as np
import scipy.signal as sig
from utility import *

def captured_sign(phi, xs, N, p):
    distance = np.sqrt((R * np.cos(phi) - xs[0]) ** 2 + (R * np.sin(phi) - xs[1]) ** 2)
    delay = distance / c
    waveform, shift, offset = fractional_delay(delay, Lf, fs=fs, type='lagrange')  # getting impulse_respones

    # getting captured signal for each microphone
    s = captured_signal(waveform, shift, p)
    return s



# Constants
c = 343  # speed of sound [m/s]
fs = 8000  # sampling frequency [Hz]

# Parameters
N = 150  # length of the impulse response
K = 90  # desired number of impulse responses
Lf = 13  # length of the fractional delay filter


# Source position
num_source = 2
xs = [[0, 2], [0, -2]]


#select difference of phase between mics
init_delta_phi = 10
num_mic = 2

# Receiver positions on a circle
R = 0.5  # radius
Q = 12
Omega = 2 * np.pi / Q
L = int(2 * np.pi / Omega * fs)
t = (1 / fs) * np.arange(L)
phi = np.zeros((num_mic, len(t)))
phi[0, :] = Omega * t
phi[1, :] = phi[0,:] + init_delta_phi
Phi = np.linspace(0, 2*np.pi, num=K, endpoint=False)

p = perfect_sweep(N)
s = np.zeros((num_source, len(phi[0])))
impulse_response = np.zeros((num_mic, N, K))

for ii in range(num_mic):
    for i in range(len(xs)):
        s[i,:] = captured_sign(phi[ii, :], xs[i], N, p)
    s = s[0, :] + s[1, :]

    #for each subsignal

    for k in range(K):
        y = np.zeros(N)
        for i in range(N):
            s_i = s[i::N]
            phi_i = phi[ii, i::N]  # Decompose the captured signal into N sub-signals
            y[i] = spatial_interpolation(s_i, phi_i, Phi[k], 'linear')  # interpolation

        # calculating of impulse_response
        impulse_response[ii, :, k] = cxcorr(y, p)
    plt.imshow(impulse_response[ii, :, :])
    plt.show()


print("end")













    ##################################################################