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
from tkinter import *

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
num_methods = 4
# Source position
num_source = 2
xs = [[0, 2], [0, -2]]
R = 0.5  # radius


top = Tk()
var1 = IntVar()
c = Checkbutton(top, text="All interpolation methods", variable=var1)
c.pack()

L1 = Label(top, text="Omega")
L1.pack(side=TOP)
textBox = Text(top, height=2, width=10)
textBox.pack()

MODES = [
    ("linear", 'linear'),
    ("NearestNeighbour", 'nearestNeighbour'),
    ("spline", 'spline'),
    ("sinc", 'sinc'),
]

v = StringVar()
v.set("L")  # initialize

for text, mode in MODES:
    b = Radiobutton(top, text=text,
                    variable=v, value=mode)
    b.pack(anchor=W)


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

    for k in range(K):
        y = np.zeros(N)
        for i in range(N):
            s_i_linear = s[i::N]
            phi_i_linear = phi[i::N]  # Decompose the captured signal into N sub-signals
            y[i] = spatial_interpolation(s_i_linear, phi_i_linear, Phi[k], interp_method)  # interpolation

        impulse_response[:, k] = cxcorr(y, p)
    DD = np.zeros(K)
    for psi in range(K):
        nummer = numerator(impulse_response[:, psi], h1[:, psi], h2[:, psi])  # numerator of formula
        denom = denominator(h1[:, psi], h2[:, psi])
        DD[psi] = nummer / denom

    return DD, impulse_response


##########################################################################################

def callback(Q, mode):
    num_mic = 2
    D1 = np.zeros((num_methods, K))
    D2 = np.zeros((num_methods, K))
    Avg_D = np.zeros((1, K))
    Phi = np.zeros((2, K))
    Phi[0, :] = np.linspace(0, 2 * np.pi, num=K, endpoint=False)
    Phi[1, :] = np.roll(Phi[0,:], int(K/2))

    #######################End of Static response######################


    # Excitation by perfet sequences.
    # p = perfect_sequence_randomphase(N)
    p = perfect_sweep(N)


    for jj in range(num_mic):
        impulse_response1 = np.zeros((num_methods, N, K))
        impulse_response2 = np.zeros((num_methods, N, K))
        #impulse_response1 = np.zeros((num_methods, N, K))
        #for ii in range(num_source):
        p1 = np.roll(p, int(N/2))
        distance = np.sqrt((R * np.cos(Phi[jj,:]) - xs[0][0]) ** 2 + (R * np.sin(Phi[jj,:]) - xs[0][1]) ** 2)
        delay = distance / cc
        weight = 1 / distance

        #######################Static impulse respones########################
        waveform, shift, _ = fractional_delay(delay, Lf, fs=fs, type='lagrange')
        h1, _, _ = construct_ir_matrix(waveform * weight[:, np.newaxis], shift, N)
        h1 = h1.T

        distance = np.sqrt((R * np.cos(Phi[jj, :]) - xs[1][0]) ** 2 + (R * np.sin(Phi[jj, :]) - xs[1][1]) ** 2)
        delay = distance / cc
        weight = 1 / distance

        #######################Static impulse respones########################
        waveform, shift, _ = fractional_delay(delay, Lf, fs=fs, type='lagrange')
        h2, _, _ = construct_ir_matrix(waveform * weight[:, np.newaxis], shift, N)
        h2 = h2.T
        # denom = denominator(h, Phi)#  denominator of formula

        s_0, phi = initialize(R, cc, p, Q, fs, Lf, xs[0])
        s_1, _ = initialize(R, cc, p1, Q, fs, Lf, xs[1])
        #s = (s_0 + s_1)

        #####################################Interpolation method is linear#####################################################
        interp_method = mode
        D1[0, :], impulse_response1[0, :] = calc_impulse_response(K, N, s_0, phi, Phi[jj,:], interp_method, h1, h2, p)
        D2[0, :], impulse_response2[0, :] = calc_impulse_response(K, N, s_1, phi, Phi[jj, :], interp_method, h1, h2, p)
        
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
        if jj==0:
            plt.title('Impulse_Response for mic 1(linear)')
        else:
            plt.title('Impulse_Response for mic 2(linear)')

        plt.imshow(db(impulse_response1[0, :]), extent=[0, K, 0, N], aspect="auto")
        plt.imshow(db(impulse_response1[0, :]+h1), extent=[0, K, 0, N], aspect="auto")
        plt.imshow(db(impulse_response2[0, :]), extent=[0, K, 0, N], aspect="auto")
        plt.imshow(db(impulse_response2[0, :]+h2), extent=[0, K, 0, N], aspect="auto")
        xx = np.linspace(0, 2*np.pi, num=90, endpoint=False)

        plt.plot(xx, db(D1[0, :]), label=mode)
        plt.plot(xx, db(D2[0, :]), label=mode)
       
        # plt.plot(Omega_seq[0, :], y_val, label="Omega_C(Q=1.375):{}".format(Qmega_o)+"rad/s")
        plt.legend()
        plt.grid()

        # plt.xlim(0, 360)
        # plt.ylim(max_o)

        plt.show()


def callback_all(Q):
    num_mic = 2
    D1 = np.zeros((num_methods, K))
    D2 = np.zeros((num_methods, K))
    D = np.zeros((num_methods, K))
    Avg_D = np.zeros((num_source, K))
    Phi = np.zeros((2, K))
    Phi[0, :] = np.linspace(0, 2 * np.pi, num=K, endpoint=False)
    Phi[1, :] = np.roll(Phi[0,:], int(K/2))

    #######################End of Static response######################


    # Excitation by perfet sequences.
    # p = perfect_sequence_randomphase(N)
    p = perfect_sweep(N)


    for jj in range(num_mic):
        impulse_response1 = np.zeros((num_methods, N, K))
        impulse_response2 = np.zeros((num_methods, N, K))
        impulse_response = np.zeros((num_methods, N, K))
        p1 = np.roll(p, int(N/2))
        distance = np.sqrt((R * np.cos(Phi[jj,:]) - xs[0][0]) ** 2 + (R * np.sin(Phi[jj,:]) - xs[0][1]) ** 2)
        delay = distance / cc
        weight = 1 / distance

        #######################Static impulse respones########################
        waveform, shift, _ = fractional_delay(delay, Lf, fs=fs, type='lagrange')
        h1, _, _ = construct_ir_matrix(waveform * weight[:, np.newaxis], shift, N)
        h1 = h1.T

        distance = np.sqrt((R * np.cos(Phi[jj, :]) - xs[1][0]) ** 2 + (R * np.sin(Phi[jj, :]) - xs[1][1]) ** 2)
        delay = distance / cc
        weight = 1 / distance

        #######################Static impulse respones########################
        waveform, shift, _ = fractional_delay(delay, Lf, fs=fs, type='lagrange')
        h2, _, _ = construct_ir_matrix(waveform * weight[:, np.newaxis], shift, N)
        h2 = h2.T
        # denom = denominator(h, Phi)#  denominator of formula

        s_0, phi = initialize(R, cc, p, Q, fs, Lf, xs[0])#
        s_1, _ = initialize(R, cc, p1, Q, fs, Lf, xs[1])
        s = (s_0 + s_1)

        #####################################Interpolation method is linear#####################################################

        D[0, :], impulse_response[0, :] = calc_impulse_response(K, N, s, phi, Phi[jj,:], 'linear', h1, h2, p)
        D[1, :], impulse_response[1, :] = calc_impulse_response(K, N, s, phi, Phi[jj,:], 'spline', h1, h2, p)
        D[2, :], impulse_response[2, :] = calc_impulse_response(K, N, s, phi, Phi[jj,:], 'nearestNeighbour', h1, h2, p)
        D[3, :], impulse_response[3, :] = calc_impulse_response(K, N, s, phi, Phi[jj,:], 'sinc', h1, h2, p)

        D1[0, :], impulse_response1[0, :] = calc_impulse_response(K, N, s_0, phi, Phi[jj, :], 'linear', h1, h2, p)
        D1[1, :], impulse_response1[1, :] = calc_impulse_response(K, N, s_0, phi, Phi[jj, :], 'spline', h1, h2, p)
        D1[2, :], impulse_response1[2, :] = calc_impulse_response(K, N, s_0, phi, Phi[jj, :], 'nearestNeighbour', h1, h2, p)
        D1[3, :], impulse_response1[3, :] = calc_impulse_response(K, N, s_0, phi, Phi[jj, :], 'sinc', h1, h2, p)

        D2[0, :], impulse_response2[0, :] = calc_impulse_response(K, N, s_1, phi, Phi[jj, :], 'linear', h1, h2, p)
        D2[1, :], impulse_response2[1, :] = calc_impulse_response(K, N, s_1, phi, Phi[jj, :], 'spline', h1, h2, p)
        D2[2, :], impulse_response2[2, :] = calc_impulse_response(K, N, s_1, phi, Phi[jj, :], 'nearestNeighbour', h1, h2, p)
        D2[3, :], impulse_response2[3, :] = calc_impulse_response(K, N, s_1, phi, Phi[jj, :], 'sinc', h1, h2, p)


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
        plt.figure()
        if jj == 0:
            plt.title('Impulse_Response for mic 1(sinc)')
        else:
            plt.title('Impulse_Response for mic 2(sinc)')
        plt.imshow(impulse_response[3, :], extent=[0, K, 0, N], aspect="auto")

        plt.figure()
        plt.xlabel(r'$\phi$/radian')
        plt.ylabel(r'$System$ $distance$ / dB')
        if jj == 0:
            plt.title('System distance for mic 1')
        else:
            plt.title('System distance for mic 2')
        xx = np.linspace(0, 2*np.pi, num=90, endpoint=False)

        plt.plot(xx, db(D[0, :]), label='linear')
        plt.plot(xx, db(D[1, :]), label='spline')
        plt.plot(xx, db(D[2, :]), label='nearestNeighbour')
        plt.plot(xx, db(D[3, :]), label='sinc')
        # plt.plot(Omega_seq[0, :], y_val, label="Omega_C(Q=1.375):{}".format(Qmega_o)+"rad/s")
        plt.legend()
        plt.grid()

        # plt.xlim(0, 360)
        # plt.ylim(max_o)

        plt.show()

def callback_avg_D():
    m_omega = 3
    Q = np.linspace(1, 10, num=m_omega, endpoint=False)  # 12
    num_mic = 2
    D = np.zeros((m_omega, num_methods, K))
    Avg_D = np.zeros((num_methods, m_omega))

    # Receiver positions on a circle

    Phi = np.linspace(0, 2 * np.pi, num=K, endpoint=False)
    distance = np.sqrt((R * np.cos(Phi) - xs[0][0]) ** 2 + (R * np.sin(Phi) - xs[0][1]) ** 2)
    delay = distance / cc
    weight = 1 / distance

    #######################Static impulse respones########################
    waveform, shift, _ = fractional_delay(delay, Lf, fs=fs, type='lagrange')
    h1, _, _ = construct_ir_matrix(waveform * weight[:, np.newaxis], shift, N)
    h1 = h1.T
    # denom = denominator(h, Phi)#  denominator of formula

    Phi = np.linspace(0, 2 * np.pi, num=K, endpoint=False)
    distance = np.sqrt((R * np.cos(Phi) - xs[1][0]) ** 2 + (R * np.sin(Phi) - xs[1][1]) ** 2)
    delay = distance / cc
    weight = 1 / distance

    #######################Static impulse respones########################
    waveform, shift, _ = fractional_delay(delay, Lf, fs=fs, type='lagrange')
    h2, _, _ = construct_ir_matrix(waveform * weight[:, np.newaxis], shift, N)
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
            D[ii, 0, :], hl = calc_impulse_response(K, N, s, phi, Phi, interp_method, h1, h2, p)
            # plt.imshow(impulse_response_linear)
            Avg_D[0, ii] = db(np.mean(D[ii, 0, :]))

            #####################################Interpolation method is nearestNeighbour#####################################################
            interp_method = 'nearestNeighbour'
            D[ii, 1, :], hn = calc_impulse_response(K, N, s, phi, Phi, interp_method, h1, h2, p)
            Avg_D[1, ii] = db(np.mean(D[ii, 1, :]))

            #####################################Interpolation method is sinc#####################################################
            interp_method = 'sinc'
            D[ii, 2, :], hs = calc_impulse_response(K, N, s, phi, Phi, interp_method, h1, h2, p)
            Avg_D[2, ii] = db(np.mean(D[ii, 2, :]))

            #####################################Interpolation method is spline#####################################################
            interp_method = 'spline'
            D[ii, 3, :], hsp = calc_impulse_response(K, N, s, phi, Phi, interp_method, h1, h2, p)
            Avg_D[3, ii] = db(np.mean(D[ii, 3, :]))

        Omega = 2 * np.pi / Q
        # Omega = np.rad2deg(2 * np.pi / Q)
        min_o = np.amin(Avg_D)
        max_o = np.amax(Avg_D)

        # Descret line for Omege_C

        y_val = np.linspace(min_o, max_o, num=50)
        Qmega_o = 2 * np.pi / 1.375
        Omega_seq = np.ones((1, 50)) * Qmega_o

        # Plot
        plt.figure()
        plt.plot(Omega, Avg_D[0, :], label="Interp_Method is linear")
        plt.plot(Omega, Avg_D[1, :], label="Interp_Method is nearestNeighbour")
        plt.plot(Omega, Avg_D[2, :], label="Interp_Method is sinc")
        plt.plot(Omega, Avg_D[3, :], label="Interp_Method is spline")
        plt.plot(Omega_seq[0, :], y_val, label="Omega_C(Q=1.375):{}".format(Qmega_o) + "rad/s")
        plt.legend()
        plt.grid()

        # plt.xlim(0, 360)
        # plt.ylim(max_o)
        plt.xlabel('Omega : rad/s')
        plt.ylabel(r'$Average$ $System$ $distance$ / dB')
        if jj == 0:
            plt.title('Average System distance for mic 1')
        if jj == 1:
            plt.title('Average System distance for mic 2')

        plt.show()


def retrieve_input():
    if var1.get() == True:
        inputValue = textBox.get("1.0", "end-1c")
        callback_all(float(inputValue))
    else:
        inputValue = textBox.get("1.0", "end-1c")
        mode = v.get()
        callback(float(inputValue), mode)


BB = Button(top, height=1, width=10, text="IR&D", command=lambda: retrieve_input())
CC = Button(top, height=1, width=10, text="Avg_D", command=lambda: callback_avg_D())

BB.pack()
CC.pack()
top.mainloop()




