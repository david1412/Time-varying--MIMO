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
cc = 343  # speed of sound [m/s]
fs = 8000  # sampling frequency [Hz]

# Parameters
N = 150  # length of the impulse response
K = 90  # desired number of impulse responses
Lf = 13  # length of the fractional delay filter
# inter_method = 4
# Q = 0.628 #[0.628, 1.375, 6.28]   #12
# m_omega = 10
# Q = np.linspace(6.28, 0.628, num=m_omega, endpoint=False)   #12
num_methods = 3
# Source position
num_source = 2
xs = [[0, 2], [0, -2]]
R = 0.5  # radius


def initialize(R, cc, p, Q, fs, Lf, xs):
    Omega = 2 * np.pi / Q  # angular speed of the microphone [rad/s]

    L = int(2 * np.pi / Omega * fs)
    t = (1 / fs) * np.arange(L)
    phi = Omega * t
    distance = np.sqrt((R * np.cos(phi) - xs[0]) ** 2 + (R * np.sin(phi) - xs[1]) ** 2)
    delay = distance / cc
    weight = 1 / distance
    type = 'lagrange'  # FD filters
    waveform, shift, offset = fractional_delay(delay, Lf, fs=fs, type=type)  # getting impulse_respones
    waveform = waveform * weight[:, np.newaxis]
    # h, _, _ = construct_ir_matrix(waveform*weight[:, np.newaxis], shift, N)
    # getting captured signal for each microphone
    s = captured_signal(waveform, shift, p)
    return s, phi


def calc_impulse_response(K, N, s, phi, Phi, interp_method, h1, h2, p):
    # calculating of impulse_response
    impulse_response = np.zeros((N, K))
    impulse_response = np.zeros((N, K))
    ir1 = np.zeros((N, K))
    ir2 = np.zeros((N, K))


    for k in range(K):
        y = np.zeros(N)
        for i in range(N):
            s_i_linear = s[i::N]
            phi_i_linear = phi[i::N]  # Decompose the captured signal into N sub-signals
            y[i] = spatial_interpolation(s_i_linear, phi_i_linear, Phi[k], interp_method)  # interpolation

        impulse_response[:, k] = cxcorr(y, p)
        ir1 = impulse_response[:int(N/2), :]
        ir2 = impulse_response[int(N/2):, :]#*2
    D1 = np.zeros(K)
    D2 = np.zeros(K)
    
    for psi in range(K):
        nummer1 = numerator(ir1[:, psi], h1[:, psi])
        denom1 = denominator(h1[:,  psi])
        D1[psi] = nummer1/denom1
        
        nummer2 = numerator(ir2[:, psi], h2[:, psi])  # numerator of formula
        denom2 = denominator(h2[:, psi])
        D2[psi] = nummer2/ denom2

    return D1, D2, ir1, ir2


##########################################################################################

def callback(Q, mode):
    num_mic = 2
    D1 = np.zeros((num_methods, K))
    D2 = np.zeros((num_methods, K))
    Avg_D = np.zeros((1, K))
    Phi = np.zeros((2, K))
    Phi[0, :] = np.linspace(0, 2 * np.pi, num=K, endpoint=False)
    Phi[1, :] = Phi[0, :]#np.roll(Phi[0,:], int(K/2))
    impulse_response1 = np.zeros((num_methods, int(N / 2), K))
    impulse_response2 = np.zeros((num_methods, int(N / 2), K))
    #######################End of Static response######################


    # Excitation by perfet sequences.
    # p = perfect_sequence_randomphase(N)
    p = perfect_sweep(N)


    for jj in range(num_mic):

        jj = 1
        #impulse_response1 = np.zeros((num_methods, N, K))
        #for ii in range(num_source):
        p1 = np.roll(p, int(N/2))
        distance = np.sqrt((R * np.cos(Phi[jj, :]) - xs[0][0]) ** 2 + (R * np.sin(Phi[jj, :]) - xs[0][1]) ** 2)
        delay = distance / cc
        weight = 1 / distance

        #######################Static impulse respones########################
        waveform, shift, _ = fractional_delay(delay, Lf, fs=fs, type='lagrange')
        h1, _, _ = construct_ir_matrix(waveform * weight[:, np.newaxis], shift, int(N/2))
        h1 = h1.T

        distance = np.sqrt((R * np.cos(Phi[jj, :]) - xs[1][0]) ** 2 + (R * np.sin(Phi[jj, :]) - xs[1][1]) ** 2)
        delay = distance / cc
        weight = 1 / distance

        #######################Static impulse respones########################
        waveform, shift, _ = fractional_delay(delay, Lf, fs=fs, type='lagrange')
        h2, _, _ = construct_ir_matrix(waveform * weight[:, np.newaxis], shift, int(N/2))
        h2 = h2.T
        # denom = denominator(h, Phi)#  denominator of formula

        s_0, phi = initialize(R, cc, p, Q, fs, Lf, xs[0])
        s_1, _ = initialize(R, cc, p1, Q, fs, Lf, xs[1])
        s = (s_0 + s_1)

        #####################################Interpolation method is linear#####################################################
        interp_method = mode
        D1[0, :], D2[0, :], impulse_response1[0, :], impulse_response2[0, :] = calc_impulse_response(K, N, s, phi, Phi[jj, :], interp_method, h1, h2, p)
        #D2[0, :], impulse_response2[0, :] = calc_impulse_response(K, N, s_1, phi, Phi[jj, :], interp_method, h1, h2, p)
        
        Omega = 2 * np.pi / Q
        # Omega = np.rad2deg(2 * np.pi / Q)
        min_o = np.amin(Avg_D)
        max_o = np.amax(Avg_D)

        # Descret line for Omege_C

        y_val = np.linspace(min_o, max_o, num=50)
        Qmega_o = 2 * np.pi / 1.375
        Omega_seq = np.ones((1, 50)) * Qmega_o

        # Plot
        #impulse_response = impulse_response + impulse_response1
        #if ii == 1:

        plt.figure()


        plt.title('Impulse_Response for mic {}'.format(jj+1) +': ' + mode)


        plt.imshow(db(impulse_response1[0, :]), extent=[0, K, 0, N], aspect="auto")
        #plt.colorbar(db(impulse_response1[0, :]))
        plt.imshow(db(h1), extent=[0, K, 0, N], aspect="auto")
        plt.imshow(db(impulse_response2[0, :]), extent=[0, K, 0, N], aspect="auto")
        plt.imshow(db(h2), extent=[0, K, 0, N], aspect="auto")
        xx = np.linspace(0, 2*np.pi, num=90, endpoint=False)

        plt.figure()
        plt.plot(xx, db(D1[0, :]), label=mode + ' 1')
        plt.plot(xx, db(D2[0, :]), label=mode + ' 2')
       
        # plt.plot(Omega_seq[0, :], y_val, label="Omega_C(Q=1.375):{}".format(Qmega_o)+"rad/s")
        plt.legend()
        plt.grid()

        # plt.xlim(0, 360)
        # plt.ylim(max_o)

        plt.show()

    return impulse_response1, impulse_response2


def callback_all(Q):
    num_mic = 2
    D1 = np.zeros((num_methods, K))
    D2 = np.zeros((num_methods, K))
    #D = np.zeros((num_methods, K))
    Avg_D = np.zeros((num_source, K))
    Phi = np.zeros((2, K))
    Phi[0, :] = np.linspace(0, 2 * np.pi, num=K, endpoint=False)
    Phi[1, :] = Phi[0, :]#np.roll(Phi[0,:], int(K/2))
    impulse_response1 = np.zeros((num_methods, int(N / 2), K))
    impulse_response2 = np.zeros((num_methods, int(N / 2), K))
    #######################End of Static response######################


    # Excitation by perfet sequences.
    # p = perfect_sequence_randomphase(N)
    p = perfect_sweep(N)


    for jj in range(num_mic):

        #impulse_response = np.zeros((num_methods, N, K))
        p1 = np.roll(p, int(N/2))
        distance = np.sqrt((R * np.cos(Phi[jj,:]) - xs[0][0]) ** 2 + (R * np.sin(Phi[jj,:]) - xs[0][1]) ** 2)
        delay = distance / cc
        weight = 1 / distance

        #######################Static impulse respones########################
        waveform, shift, _ = fractional_delay(delay, Lf, fs=fs, type='lagrange')
        h1, _, _ = construct_ir_matrix(waveform * weight[:, np.newaxis], shift, int(N/2))
        h1 = h1.T

        distance = np.sqrt((R * np.cos(Phi[jj, :]) - xs[1][0]) ** 2 + (R * np.sin(Phi[jj, :]) - xs[1][1]) ** 2)
        delay = distance / cc
        weight = 1 / distance

        #######################Static impulse respones########################
        waveform, shift, _ = fractional_delay(delay, Lf, fs=fs, type='lagrange')
        h2, _, _ = construct_ir_matrix(waveform * weight[:, np.newaxis], shift, int(N/2))
        h2 = h2.T
        # denom = denominator(h, Phi)#  denominator of formula

        s_0, phi = initialize(R, cc, p, Q, fs, Lf, xs[0])#
        s_1, _ = initialize(R, cc, p1, Q, fs, Lf, xs[1])
        s = (s_0 + s_1)

        #####################################Interpolation method is linear#####################################################

        #D[0, :], impulse_response[0, :] = calc_impulse_response(K, N, s, phi, Phi[jj,:], 'linear', h1, h2, p)
        #D[1, :], impulse_response[1, :] = calc_impulse_response(K, N, s, phi, Phi[jj,:], 'spline', h1, h2, p)
        #D[2, :], impulse_response[2, :] = calc_impulse_response(K, N, s, phi, Phi[jj,:], 'nearestNeighbour', h1, h2, p)


        D1[0, :], D2[0, :], impulse_response1[0, :], impulse_response2[0, :] = calc_impulse_response(K, N, s, phi, Phi[jj, :], 'linear', h1, h2, p)
        D1[1, :], D2[1, :], impulse_response1[1, :], impulse_response2[1, :] = calc_impulse_response(K, N, s, phi, Phi[jj, :], 'spline', h1, h2, p)
        D1[2, :], D2[2, :], impulse_response1[2, :], impulse_response2[2, :] = calc_impulse_response(K, N, s, phi, Phi[jj, :], 'nearestNeighbour', h1, h2, p)


        #D2[0, :], impulse_response2[0, :] = calc_impulse_response(K, N, s_1, phi, Phi[jj, :], 'linear', h1, h2, p)
        #D2[1, :], impulse_response2[1, :] = calc_impulse_response(K, N, s_1, phi, Phi[jj, :], 'spline', h1, h2, p)
        #D2[2, :], impulse_response2[2, :] = calc_impulse_response(K, N, s_1, phi, Phi[jj, :], 'nearestNeighbour', h1, h2, p)



        Omega = 2 * np.pi / Q
        # Omega = np.rad2deg(2 * np.pi / Q)
        min_o = np.amin(Avg_D)
        max_o = np.amax(Avg_D)

        # Descret line for Omege_C

        y_val = np.linspace(min_o, max_o, num=50)
        Qmega_o = 2 * np.pi / 1.375
        Omega_seq = np.ones((1, 50)) * Qmega_o
        

        plt.figure()
        plt.imshow(impulse_response1[0, :], extent=[0, 90, 0, 150], aspect="auto")
        plt.colorbar(label='dB')
        plt.clim(-0.1,0.5)
        plt.xlabel(r'$\phi$/ Degree')
        plt.ylabel(r'Samples')
        plt.title('Impulse_Response1(linear)for mic{}'.format(jj+1))
        

        plt.figure()
        plt.imshow(h1, extent=[0, 90, 0, 150], aspect="auto")
        plt.colorbar(label='dB')
        plt.clim(-0.1,0.5)
        plt.xlabel(r'$\phi$/ Degree')
        plt.ylabel(r'Samples')
        plt.title('h1(linear)for mic{}'.format(jj+1))
        

        plt.figure()
        plt.imshow(impulse_response2[0, :], extent=[0, 90, 0, 150], aspect="auto")
        plt.colorbar(label='dB')
        plt.clim(-0.1,0.5)
        plt.xlabel(r'$\phi$/ Degree')
        plt.ylabel(r'Samples')
        plt.title('Impulse_Response2(linear)for mic{}'.format(jj+1))
        
        
        plt.figure()
        plt.imshow(h2, extent=[0, 90, 0, 150], aspect="auto")
        plt.colorbar(label='dB')
        plt.clim(-0.1,0.5)
        plt.xlabel(r'$\phi$/ Degree')
        plt.ylabel(r'Samples')
        plt.title('h2(linear)for mic{}'.format(jj+1))
        

        plt.figure()
        plt.imshow(impulse_response1[1, :], extent=[0, 90, 0, 150], aspect="auto")
        plt.colorbar(label='dB')
        plt.clim(-0.1,0.5)
        plt.xlabel(r'$\phi$/ Degree')
        plt.ylabel(r'Samples')
        plt.title('Impulse_Response1(spline)for mic{}'.format(jj+1))
        
        
        plt.figure()
        plt.imshow(h1, extent=[0, 90, 0, 150], aspect="auto")
        plt.colorbar(label='dB')
        plt.clim(-0.1,0.5)
        plt.xlabel(r'$\phi$/ Degree')
        plt.ylabel(r'Samples')
        plt.title('h1(spline)for mic{}'.format(jj+1))
        
        
        plt.figure()
        plt.imshow(impulse_response2[1, :], extent=[0, 90, 0, 150], aspect="auto")
        plt.colorbar(label='dB')
        plt.clim(-0.1,0.5)
        plt.xlabel(r'$\phi$/ Degree')
        plt.ylabel(r'Samples')
        plt.title('Impulse_Response2(spline)for mic{}'.format(jj+1))
        
        
        plt.figure()
        plt.imshow(h2, extent=[0,90 , 0, 150], aspect="auto")
        plt.colorbar(label='dB')
        plt.clim(-0.1,0.5)
        plt.xlabel(r'$\phi$/ Degree')
        plt.ylabel(r'Samples')
        plt.title('h2(spline)for mic{}'.format(jj+1))
        

        plt.figure()
        plt.imshow(impulse_response1[2, :], extent=[0, 90, 0, 150], aspect="auto")
        plt.colorbar(label='dB')
        plt.clim(-0.1,0.5)
        plt.xlabel(r'$\phi$/ Degree')
        plt.ylabel(r'Samples')
        plt.title('Impulse_Response1(NearestNeighbour)for mic{}'.format(jj+1))
        
        
        plt.figure()
        plt.imshow(h1, extent=[0, 90, 0, 150], aspect="auto")
        plt.colorbar(label='dB')
        plt.clim(-0.1,0.5)
        plt.xlabel(r'$\phi$/ Degree')
        plt.ylabel(r'Samples')
        plt.title('h1(NearestNeighbour)for mic{}'.format(jj+1))
        
        
        plt.figure()
        plt.imshow(impulse_response2[2, :], extent=[0, 90, 0, 150], aspect="auto")
        plt.colorbar(label='dB')
        plt.clim(-0.1,0.5)
        plt.xlabel(r'$\phi$/ Degree')
        plt.ylabel(r'Samples')
        plt.title('Impulse_Response2(NearestNeighbour)for mic{}'.format(jj+1))
       
        
        plt.figure()
        plt.imshow(h2, extent=[0, 90, 0, 150], aspect="auto")
        plt.colorbar(label='dB')
        plt.clim(-0.1,0.5)
        plt.xlabel(r'$\phi$/ Degree')
        plt.ylabel(r'Samples')
        plt.title('h2(NearestNeighbour)for mic{}'.format(jj+1))

        
        # Plot
        #impulse_response = impulse_response + impulse_response1
        #if ii == 1:
       #########################
        plt.figure()
        if jj==0:
            plt.title('Impulse_Response for mic 1(linear)')
        else:
            plt.title('Impulse_Response for mic 2(linear)')

        plt.imshow(impulse_response[0, :], extent=[0, K, 0, N], aspect="auto")
        plt.figure()
        if jj == 0:
            plt.title('Impulse_Response for mic 1(NearestNeighbour)')
        else:
            plt.title('Impulse_Response for mic 2(NearestNeighbour)')
        plt.imshow(impulse_response[1, :], extent=[0, K, 0, N], aspect="auto")
        plt.figure()
        if jj == 0:
            plt.title('Impulse_Response for mic 1(spline)')
        else:
            plt.title('Impulse_Response for mic 2(spline)')
        plt.imshow(impulse_response[2, :], extent=[0, K, 0, N], aspect="auto")
        
       ####################################
        plt.figure()
        plt.xlabel(r'$\phi$/radian')
        plt.ylabel(r'$System$ $distance$ / dB')
        plt.title('System distance 1 for mic {}'.format(jj+1))
        xx = np.linspace(0, 2*np.pi, num=90, endpoint=False)

        plt.plot(xx, db(D1[0, :]), label='linear')
        plt.plot(xx, db(D1[1, :]), label='spline')
        plt.plot(xx, db(D1[2, :]), label='nearestNeighbour')
        

        # plt.plot(Omega_seq[0, :], y_val, label="Omega_C(Q=1.375):{}".format(Qmega_o)+"rad/s")
        plt.legend()
        plt.grid()


        plt.figure()
        plt.xlabel(r'$\phi$/radian')
        plt.ylabel(r'$System$ $distance$ / dB')
        plt.title('System distance 2 for mic {}'.format(jj + 1))
        xx = np.linspace(0, 2 * np.pi, num=90, endpoint=False)

        plt.plot(xx, db(D2[0, :]), label='linear')
        plt.plot(xx, db(D2[1, :]), label='spline')
        plt.plot(xx, db(D2[2, :]), label='nearestNeighbour')
       
        # plt.plot(Omega_seq[0, :], y_val, label="Omega_C(Q=1.375):{}".format(Qmega_o)+"rad/s")
        plt.legend()
        plt.grid()

        # plt.xlim(0, 360)
        # plt.ylim(max_o)

        plt.show()

    return impulse_response1, impulse_response2

def callback_avg_D():
    m_omega = 3
    Q = np.linspace(1, 10, num=m_omega, endpoint=False)  # 12
    num_mic = 2
    D1 = np.zeros((m_omega, num_methods, K))
    D2 = np.zeros((m_omega, num_methods, K))
    Avg_D1 = np.zeros((num_methods, m_omega))
    Avg_D2 = np.zeros((num_methods, m_omega))

    # Receiver positions on a circle

    Phi = np.linspace(0, 2 * np.pi, num=K, endpoint=False)
    distance = np.sqrt((R * np.cos(Phi) - xs[0][0]) ** 2 + (R * np.sin(Phi) - xs[0][1]) ** 2)
    delay = distance / cc
    weight = 1 / distance

    #######################Static impulse respones########################
    waveform, shift, _ = fractional_delay(delay, Lf, fs=fs, type='lagrange')
    h1, _, _ = construct_ir_matrix(waveform * weight[:, np.newaxis], shift, int(N/2))
    h1 = h1.T
    # denom = denominator(h, Phi)#  denominator of formula

    Phi = np.linspace(0, 2 * np.pi, num=K, endpoint=False)
    distance = np.sqrt((R * np.cos(Phi) - xs[1][0]) ** 2 + (R * np.sin(Phi) - xs[1][1]) ** 2)
    delay = distance / cc
    weight = 1 / distance

    #######################Static impulse respones########################
    waveform, shift, _ = fractional_delay(delay, Lf, fs=fs, type='lagrange')
    h2, _, _ = construct_ir_matrix(waveform * weight[:, np.newaxis], shift, int(N/2))
    h2 = h2.T

    p = perfect_sweep(N)

    for jj in range(num_mic):
        for ii in range(len(Q)):
            p1 = np.roll(p, int(N / 2))
            s_0, phi = initialize(R, cc, p, Q[ii], fs, Lf, xs[0])
            s_1, _ = initialize(R, cc, p1, Q[ii], fs, Lf, xs[1])
            s = (s_0 + s_1)

            impulse_response = np.zeros((N, K))

            #####################################Interpolation method is linear#####################################################
            interp_method = 'linear'
            D1[ii, 0, :],D2[ii, 0, :], hl1, hl2 = calc_impulse_response(K, N, s, phi, Phi, interp_method, h1, h2, p)
            # plt.imshow(impulse_response_linear)
            Avg_D1[0, ii] = db(np.mean(D1[ii, 0, :]))
            Avg_D2[0, ii] = db(np.mean(D2[ii, 0, :]))

            #####################################Interpolation method is nearestNeighbour#####################################################
            interp_method = 'nearestNeighbour'
            D1[ii, 1, :], D2[ii, 1, :], hn1, hn2 = calc_impulse_response(K, N, s, phi, Phi, interp_method, h1, h2, p)
            Avg_D1[1, ii] = db(np.mean(D1[ii, 1, :]))
            Avg_D2[1, ii] = db(np.mean(D2[ii, 1, :]))

            #####################################Interpolation method is sinc#####################################################
            interp_method = 'spline'
            D1[ii, 2, :], D2[ii, 2, :], hs1,hs2= calc_impulse_response(K, N, s, phi, Phi, interp_method, h1, h2, p)
            Avg_D1[2, ii] = db(np.mean(D1[ii, 2, :]))
            Avg_D2[2, ii] = db(np.mean(D2[ii, 2, :]))



        Omega = 2 * np.pi / Q
        # Omega = np.rad2deg(2 * np.pi / Q)
        min_o = np.amin(Avg_D1)
        max_o = np.amax(Avg_D1)

        # Descret line for Omege_C

        y_val = np.linspace(min_o, max_o, num=50)
        Qmega_o = 2 * np.pi / 1.375
        Omega_seq = np.ones((1, 50)) * Qmega_o
        

        # Plot
        plt.figure()
        plt.plot(Omega, Avg_D1[0, :], label="Interp_Method is linear")
        plt.plot(Omega, Avg_D1[1, :], label="Interp_Method is nearestNeighbour")
        plt.plot(Omega, Avg_D1[2, :], label="Interp_Method is spline")

        plt.plot(Omega_seq[0, :], y_val, label="Omega_C(Q=1.375):{}".format(Qmega_o) + "rad/s")
        plt.legend()
        plt.grid()

        plt.xlabel('Omega : rad/s')
        plt.ylabel(r'$Average$ $System$ $distance$ / dB')
        if jj == 0:
            plt.title('Average System distance 1 for mic 1')
        if jj == 1:
            plt.title('Average System distance 1 for mic 2')

        plt.figure()
        plt.plot(Omega, Avg_D2[0, :], label="Interp_Method is linear")
        plt.plot(Omega, Avg_D2[1, :], label="Interp_Method is nearestNeighbour")
        plt.plot(Omega, Avg_D2[2, :], label="Interp_Method is spline")

        plt.plot(Omega_seq[0, :], y_val, label="Omega_C(Q=1.375):{}".format(Qmega_o) + "rad/s")
        plt.legend()
        plt.grid()

        # plt.xlim(0, 360)
        # plt.ylim(max_o)
        plt.xlabel('Omega : rad/s')
        plt.ylabel(r'$Average$ $System$ $distance$ / dB')
        if jj == 0:
            plt.title('Average System distance 2 for mic 1')
        if jj == 1:
            plt.title('Average System distancec 2 for mic 2')

        plt.show()

    return Avg_D1, Avg_D2


ir1, ir2 = callback(4, 'linear')# you can change parameters.
ir1_all, ir2_all = callback_all(4)
avg_D1,  avg_D2 = callback_avg_D()



