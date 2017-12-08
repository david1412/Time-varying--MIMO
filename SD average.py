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
from utility import *



    

# Constants
c = 343  # speed of sound [m/s]
fs = 8000  # sampling frequency [Hz]

# Parameters
N = 150  # length of the impulse response
K = 90  # desired number of impulse responses
Lf = 13  # length of the fractional delay filter
# Q = [0.628, 1.375, 6.28]   #12
Q = np.linspace(6.28, 0.628, num=K, endpoint=False)   #12

# Source position
xs = [0, 2]
D = np.zeros((K, 3, K))
Avg_D = np.zeros((K, K))

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
    Omega = 2 * np.pi / Q[ii]  # angular speed of the microphone [rad/s]

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
    impulse_response = np.zeros((N, K))

    #####################################Interpolation method is linear#####################################################
    interp_method = 'linear'
    #for each subsignal
    for k in range(K):
        y = np.zeros(N)
        for i in range(N):
            s_i_linear = s[i::N]
            phi_i_linear = phi[i::N] # Decompose the captured signal into N sub-signals
            #print(k)
            #print(Phi[k])
            y[i] = spatial_interpolation(s_i_linear, phi_i_linear, Phi[k], interp_method)  # interpolation

        #calculating of impulse_response
        impulse_response[:, k] = cxcorr(y, p)

     #formula

    for psi in range(K):
        nummer = numerator(impulse_response[:, psi], h[:, psi])# numerator of formula
        denom = denominator(h[:, psi])
        D[ii, 0, psi] = nummer/denom


    Avg_D[0, ii] = 10*np.log10(average_fwai(D[ii, 0, :], np.rad2deg(np.linspace(90, 270, num=K))))


    #####################################Interpolation method is nearestNeighbour#####################################################
    interp_method = 'nearestNeighbour'
    #for each subsignal
    for k in range(K):
        y = np.zeros(N)
        for i in range(N):
            s_i_nearest = s[i::N]
            phi_i_nearest = phi[i::N] # Decompose the captured signal into N sub-signals
            #print(k)
            #print(Phi[k])
            y[i] = spatial_interpolation(s_i_nearest, phi_i_nearest, Phi[k], interp_method)  # interpolation

        #calculating of impulse_response
        impulse_response[:, k] = cxcorr(y, p)

     #formula

    for psi in range(K):
        nummer = numerator(impulse_response[:, psi], h[:, psi]) # numerator of formula
        denom = denominator(h[:, psi])
        D[ii, 1, psi] = nummer/denom # 10*np.log10(nummer/denom)
    Avg_D[1,ii] = 10*np.log10(average_fwai(D[ii, 1, :], np.rad2deg(np.linspace(90, 270, num=K))))


    #####################################Interpolation method is sinc#####################################################
    interp_method = 'sinc'
    #for each subsignal
    for k in range(K):
        y = np.zeros(N)
        for i in range(N):
            s_i_sinc = s[i::N]
            phi_i_sinc = phi[i::N] # Decompose the captured signal into N sub-signals
            y[i] = spatial_interpolation(s_i_sinc, phi_i_sinc, Phi[k], interp_method)  #interpolation

        #calculating of impulse_response
        impulse_response[:,k] = cxcorr(y, p)

     #formula

    for psi in range(K):
        nummer = numerator(impulse_response[:, psi], h[:, psi]) # numerator of formula
        denom = denominator(h[:, psi])
        D[ii, 2,psi] = nummer/denom
    Avg_D[2, ii] = 20*np.log10(average_fwai(D[ii, 2, :], np.rad2deg(np.linspace(90, 270, num=K))))
    #######################################################################################################


Omega = np.rad2deg(2 * np.pi / Q)
min_o = np.amin(Avg_D)
max_o = np.amax(Avg_D)

y_val = np.linspace(min_o, max_o, num=50)

Qmega_o = np.rad2deg(2 * np.pi / 1.375)
Omega_seq = np.ones((1, 50)) * Qmega_o

# Plot
plt.figure()
plt.plot(Omega, Avg_D[0, :], label="linear")
plt.plot(Omega, Avg_D[1, :], label="nearestNeighbour")
plt.plot(Omega, Avg_D[2, :], label="sinc")
plt.plot(Omega_seq[0, :], y_val, label="Omega0")
plt.legend()
plt.grid()

#plt.xlim(0, 360)
#plt.ylim(max_o)
plt.xlabel(r'$\Omega$ : deg/s')
plt.ylabel(r'Average System distance / dB')

plt.title('Average System distance')
plt.show()






