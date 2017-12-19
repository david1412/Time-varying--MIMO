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

def captured_sign(phase_i, xs, N, perfect_s):
    distance = np.sqrt((R * np.cos(phase_i) - xs[0]) ** 2 + (R * np.sin(phase_i) - xs[1]) ** 2)
    delay = distance / c
    waveform, shift, offset = fractional_delay(delay, Lf, fs=fs, type='lagrange')  # getting impulse_respones

    # getting captured signal for each microphone
    signal = captured_signal(waveform, shift, perfect_s)
    return signal



# Constants
c = 343  # speed of sound [m/s]
fs = 8000  # sampling frequency [Hz]

# Parameters
N = 150  # length of the impulse response
K = 90  # desired number of impulse responses
Lf = 13  # length of the fractional delay filter


# Source position
xs = [[0, 2], [0, -2]]


# Receiver positions on a circle
R = 0.5  # radius
Q = 12
Omega = 2 * np.pi / Q
L = int(2 * np.pi / Omega * fs)
t = (1 / fs) * np.arange(L)
phi = Omega * t
Phi = np.linspace(0, 2*np.pi, num=K, endpoint=False)

p = perfect_sweep(N)

s = np.zeros((2, len(phi)))
for i in range(len(xs)):
    s[i,:] = captured_sign(phi, xs[i], N, p)

#s = s[0, :] + s[1, :]

impulse_response = np.zeros((N, K))
D = np.zeros((len(xs), K))
#for each subsignal
for ii in range(len(xs)):
    for k in range(K):
        y = np.zeros(N)
        for i in range(N):
            s_i = s[ii, i::N]
            phi_i = phi[i::N]  # Decompose the captured signal into N sub-signals
            y[i] = spatial_interpolation(s_i, phi_i, Phi[k], 'linear')  # interpolation

        # calculating of impulse_response
        impulse_response[:, k] = cxcorr(y, p)


    #######################Static impulse respones########################
    distance = np.sqrt((R * np.cos(phi) - xs[ii][0]) ** 2 + (R * np.sin(phi) - xs[ii][1]) ** 2)
    delay = distance / c
    weight = 1 / distance
    waveform, shift, _ = fractional_delay(delay, Lf, fs=fs, type='lagrange')
    h, _, _ = construct_ir_matrix(waveform * weight[:, np.newaxis], shift, N)
    h = h.T
    # denom = denominator(h, Phi)#  denominator of formula
    #######################End of Static response######################


    for psai in range(K):
        nummer = numerator(impulse_response[:, psai], h[:, psai])#numerator of formula
        denom = denominator(h[:, psai])
        D[ii, psai] = 10*np.log10(nummer/denom)

    plt.imshow(impulse_response)
    plt.show()
    plt.imshow(D)
    plt.show()
print("end")













    ##################################################################
