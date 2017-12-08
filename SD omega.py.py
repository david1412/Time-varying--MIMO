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
Q = [0.628, 1.375, 6.28]   #12
K = 90  # desired number of impulse responses
Lf = 13  # length of the fractional delay filter

# Source position
xs = [0, 2]
D = np.zeros((3, 3, K))
Avg_D = np.zeros((3, 3, K))

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
#p = perfect_sequence_randomphase(N)
p = perfect_sweep(N)

for ii in range(len(Q)):
    Omega = 2 * np.pi / Q[ii]  # angular speed of the microphone [rad/s]

    L = int(2 * np.pi / Omega * fs)
    t = (1 / fs) * np.arange(L)
    phi = Omega * t
    distance = np.sqrt((R*np.cos(phi)-xs[0])**2 + (R*np.sin(phi)-xs[1])**2)
    delay = distance / c
    weight= 1/ distance
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
            phi_i_linear = phi[i::N] #Decompose the captured signal into N sub-signals
            #print(k)
            #print(Phi[k])
            y[i] = spatial_interpolation(s_i_linear, phi_i_linear, Phi[k], interp_method)  #interpolation

        #calculating of impulse_response
        impulse_response[:,k] = cxcorr(y, p)

     #formula

    for psi in range(K):
        nummer = numerator(impulse_response[:, psi], h[:, psi])#numerator of formula
        denom = denominator(h[:, psi])
        D[ii,0,psi] = 10*np.log10(nummer/denom)


    #######################################################################################################

    #####################################Interpolation method is nearestNeighbour#####################################################
    interp_method = 'nearestNeighbour'
    #for each subsignal
    for k in range(K):
        y = np.zeros(N)
        for i in range(N):
            s_i_nearest = s[i::N]
            phi_i_nearest = phi[i::N] #Decompose the captured signal into N sub-signals
            #print(k)
            #print(Phi[k])
            y[i] = spatial_interpolation(s_i_nearest, phi_i_nearest, Phi[k], interp_method)  #interpolation

        #calculating of impulse_response
        impulse_response[:,k] = cxcorr(y, p)

     #formula

    for psi in range(K):
        nummer = numerator(impulse_response[:,psi],h[:,psi])#numerator of formula
        denom = denominator(h[:,psi])
        D[ii, 1,psi] = 10*np.log10(nummer/denom)

    #######################################################################################################

    #####################################Interpolation method is sinc#####################################################
    interp_method = 'sinc'
    #for each subsignal
    for k in range(K):
        y = np.zeros(N)
        for i in range(N):
            s_i_sinc = s[i::N]
            phi_i_sinc = phi[i::N] #Decompose the captured signal into N sub-signals
            #print(k)
            #print(Phi[k])
            y[i] = spatial_interpolation(s_i_sinc, phi_i_sinc, Phi[k], interp_method)  #interpolation

        #calculating of impulse_response
        impulse_response[:,k] = cxcorr(y, p)

     #formula

    for psi in range(K):
        nummer = numerator(impulse_response[:,psi],h[:,psi])#numerator of formula
        denom = denominator(h[:,psi])
        D[ii, 2,psi] = 10*np.log10(nummer/denom)

    #######################################################################################################
Phi=np.rad2deg(Phi)
for ii in range(3):

    # Plot
    plt.figure()
    plt.plot(Phi, D[ii, 0,:],label = "is linear")
    plt.plot(Phi, D[ii, 1,:],label = "nearestNeighbour")
    plt.plot(Phi, D[ii, 2,:],label = "sinc")
    plt.legend()
    plt.grid()
    plt.ylim(-70,10)

    #plt.xlim(0, 360)
    plt.xlabel(r'$\varphi$ / deg')
    plt.ylabel(r'System$ $distance / dB')
    if ii == 0:
        title = (r'$\Omega$ $10rad$/s')
    elif ii==1:
        title = (r'$\Omega$ $4.57rad$/s')
    else:
        title = (r'$\Omega$ $1rad$/s')
    plt.title(title)
    plt.show()











