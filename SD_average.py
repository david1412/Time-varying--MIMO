# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 13:55:01 2017

@author: davidkumar
"""

import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf, InterpolatedUnivariateSpline
from scipy import interpolate
from utility import*

def initialize(Q, fs, Lf):
    Omega = 2 * np.pi / Q  # angular speed of the microphone [rad/s]

    L = int(2 * np.pi / Omega * fs)
    t = (1 / fs) * np.arange(L)
    phi = Omega * t
    distance = np.sqrt((R*np.cos(phi)-xs[0])**2 + (R*np.sin(phi)-xs[1])**2)
    delay = distance / c
    weight = 1 / distance
    type = 'lagrange'  # FD filters
    waveform, shift, offset = fractional_delay(delay, Lf, fs=fs, type=type) # getting impulse_respones
    waveform = waveform * weight[:, np.newaxis]
    #h, _, _ = construct_ir_matrix(waveform*weight[:, np.newaxis], shift, N)
    # getting captured signal for each microphone
    s = captured_signal(waveform, shift, p)
    return s, phi


def calc_impulse_response(K, N, s, phi, Phi, interp_method, h, p):
    # calculating of impulse_response
    impulse_response = np.zeros((N, K))

    for k in range(K):
        y = np.zeros(N)
        for i in range(N):
            s_i_linear = s[i::N]
            phi_i_linear = phi[i::N] # Decompose the captured signal into N sub-signals
            y[i] = spatial_interpolation(s_i_linear, phi_i_linear, Phi[k], interp_method)  # interpolation


        impulse_response[:, k] = cxcorr(y, p)
    DD = np.zeros(K)
    for psi in range(K):
        nummer = numerator(impulse_response[:, psi], h[:, psi])# numerator of formula
        denom = denominator(h[:, psi])
        DD[psi] = nummer/denom

    return DD, impulse_response
##########################################################################################

# Constants
c = 343  # speed of sound [m/s]
fs = 8000  # sampling frequency [Hz]

# Parameters
N = 150  # length of the impulse response
K = 90  # desired number of impulse responses
Lf = 13  # length of the fractional delay filter
inter_method = 4
# Q = [0.628, 1.375, 6.28]   #12
m_omega = 10
Q = np.linspace(6.28, 0.628, num=m_omega, endpoint=False)   #12

# Source position
xs = [0, 2]
D = np.zeros((m_omega, inter_method, K))
Avg_D = np.zeros((inter_method, m_omega))

# Receiver positions on a circle
R = 0.5  # radius
Phi = np.linspace(0, 2*np.pi, num=K, endpoint=False)
distance = np.sqrt((R*np.cos(Phi)-xs[0])**2 + (R*np.sin(Phi)-xs[1])**2)
delay = distance / c
weight = 1 / distance

#######################Static impulse respones########################
waveform, shift, _ = fractional_delay(delay, Lf, fs=fs, type='lagrange')
h, _, _ = construct_ir_matrix(waveform*weight[:, np.newaxis], shift, N)
h = h.T
#denom = denominator(h, Phi)#  denominator of formula

#######################End of Static response######################


# Excitation by perfet sequences.
# p = perfect_sequence_randomphase(N)
p = perfect_sweep(N)




for ii in range(len(Q)):

    s, phi = initialize(Q[ii], fs, Lf)
    impulse_response = np.zeros((N, K))

    #####################################Interpolation method is linear#####################################################
    interp_method = 'linear'
    D[ii, 0, :], _ = calc_impulse_response(K, N, s, phi, Phi, interp_method, h, p)
    Avg_D[0, ii] = 20*np.log10(average_fwai(D[ii, 0, :], np.linspace(90, 270, num=K)))

    #####################################Interpolation method is nearestNeighbour#####################################################
    interp_method = 'nearestNeighbour'
    D[ii, 1, :], _ = calc_impulse_response(K, N, s, phi, Phi, interp_method, h, p)
    Avg_D[1, ii] = 20 * np.log10(average_fwai(D[ii, 1, :], np.linspace(90, 270, num=K)))

    #####################################Interpolation method is sinc#####################################################
    interp_method = 'sinc'
    D[ii, 2, :], _ = calc_impulse_response(K, N, s, phi, Phi, interp_method, h, p)
    Avg_D[2, ii] = 20 * np.log10(average_fwai(D[ii, 2, :], np.linspace(90, 270, num=K)))

    #####################################Interpolation method is spline#####################################################
    interp_method = 'spline'
    D[ii, 3, :], _ = calc_impulse_response(K, N, s, phi, Phi, interp_method, h, p)
    Avg_D[3, ii] = 20 * np.log10(average_fwai(D[ii, 3, :], np.linspace(90, 270, num=K)))


Omega = 2 * np.pi / Q
#Omega = np.rad2deg(2 * np.pi / Q)
min_o = np.amin(Avg_D)
max_o = np.amax(Avg_D)

#Descret line for Omege_C

y_val = np.linspace(min_o, max_o, num=50)
Qmega_o = 2 * np.pi / 1.375
Omega_seq = np.ones((1, 50)) * Qmega_o

# Plot
plt.figure()
plt.plot(Omega, Avg_D[0, :], label="linear")
plt.plot(Omega, Avg_D[1, :], label="nearestNeighbour")
plt.plot(Omega, Avg_D[2, :], label="sinc")
plt.plot(Omega, Avg_D[3, :], label="spline")
plt.plot(Omega_seq[0, :], y_val, label="Omega_C(Q=1.375):{}".format(Qmega_o)+"rad/s")
plt.legend()
plt.grid()

#plt.xlim(0, 360)
#plt.ylim(max_o)
plt.xlabel('Omega : rad/s')
plt.ylabel(r'$Average$ $System$ $distance$ / dB')
plt.title('Average System distance')
plt.show()






