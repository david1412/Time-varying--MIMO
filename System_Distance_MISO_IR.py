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

cc = 343  # speed of sound [m/s]
fs = 8000  # sampling frequency [Hz]

# Parameters
N = 150  # length of the impulse response
K = 90  # desired number of impulse responses
Lf = 13  # length of the fractional delay filter
R = 0.5  # radius

top = Tk()
var1 = IntVar()
c = Checkbutton(top, text="All interpolation methods", variable=var1)
c.pack()

L1 = Label(top, text="Omega")
L1.pack(side=TOP)
textBox=Text(top, height=2, width=10)
textBox.pack()

MODES = [
        ("linear", 'linear'),
        ("NearestNeighbour", 'nearestNeighbour'),
        ("spline", 'spline'),
        ("sinc", 'sinc'),
    ]

v = StringVar()
v.set("L") # initialize

for text, mode in MODES:
    b = Radiobutton(top, text=text,
                        variable=v, value=mode)
    b.pack(anchor=W)



def initialize(Q, fs, Lf, xs, p):
    #print(xs)
    Omega = 2 * np.pi / Q  # angular speed of the microphone [rad/s]

    L = int(2 * np.pi / Omega * fs)
    t = (1 / fs) * np.arange(L)
    phi = Omega * t
    distance = np.sqrt((R*np.cos(phi)-xs[0])**2 + (R*np.sin(phi)-xs[1])**2)
    delay = distance / cc
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

def callback(Q, mode):
    # Constants

    #inter_method = 4
    #Q = 0.628 #[0.628, 1.375, 6.28]   #12
    #m_omega = 10
    #Q = np.linspace(6.28, 0.628, num=m_omega, endpoint=False)   #12


    # Source position
    num_source = 2
    xs = [[0, 2], [0, -2]]
    D = np.zeros((num_source, K))

    Avg_D = np.zeros((num_source, K))

    # Receiver positions on a circle
    R = 0.5  # radius
    Phi = np.linspace(0, 2*np.pi, num=K, endpoint=False)


    #######################End of Static response######################


    # Excitation by perfet sequences.
    # p = perfect_sequence_randomphase(N)
    p = perfect_sweep(N)

    impulse_response = np.zeros((N, K))


    #for ii in range(num_source):

    distance = np.sqrt((R * np.cos(Phi) - xs[0][0]) ** 2 + (R * np.sin(Phi) - xs[0][1]) ** 2)
    delay = distance / cc
    weight = 1 / distance

    #######################Static impulse respones########################
    waveform, shift, _ = fractional_delay(delay, Lf, fs=fs, type='lagrange')
    h, _, _ = construct_ir_matrix(waveform * weight[:, np.newaxis], shift, N)
    h = h.T
    # denom = denominator(h, Phi)#  denominator of formula

    s_0, phi = initialize(Q, fs, Lf, xs[0], p)
    # print('s_0:  ')
    # print(s_0)
    s_1, _ = initialize(Q, fs, Lf, xs[1], np.roll(p, int(N / 2)))
    # print('s_1:  ')
    # print(s_1)
    s = (s_0 + s_1)

    #####################################Interpolation method is linear#####################################################
    interp_method = mode
    D[0, :], impulse_response = calc_impulse_response(K, N, s, phi, Phi, interp_method, h, p)
    #Avg_D[0, ii] = 20*np.log10(average_fwai(D[ii, 0, :], np.linspace(90, 270, num=K)))


    Omega = 2 * np.pi / Q
    #Omega = np.rad2deg(2 * np.pi / Q)
    min_o = np.amin(Avg_D)
    max_o = np.amax(Avg_D)

    #Descret line for Omege_C

    y_val = np.linspace(min_o, max_o, num=50)
    Qmega_o = 2 * np.pi / 1.375
    Omega_seq = np.ones((1, 50)) * Qmega_o

    # Plot

    #impulse_response += impulse_response1
    #if ii==1:
    plt.figure()
    plt.xlabel(r'$\phi$ /  ⁰')
    plt.ylabel(r'$Impulse$ $Response$ / dB')
    plt.imshow(impulse_response, extent=[0, 360, 0, 150], aspect="auto")


    plt.figure()
    xx = np.linspace(0, 2*np.pi, num = 90, endpoint=False)
    plt.plot(xx, db(D[0]))
    #plt.plot(Omega_seq[0, :], y_val, label="Omega_C(Q=1.375):{}".format(Qmega_o)+"rad/s")
    plt.legend()
    plt.grid()

    #plt.xlim(0, 360)
    #plt.ylim(max_o)
    plt.xlabel(r'$\phi$/  radian')
    plt.ylabel(r'$System$ $distance$ / dB')
    plt.title('System distance')
    plt.show()


def callback_all(Q):

    #inter_method = 4
    #Q = 0.628 #[0.628, 1.375, 6.28]   #12
    #m_omega = 10
    #Q = np.linspace(6.28, 0.628, num=m_omega, endpoint=False)   #12

    num_methods = 4
    # Source position
    num_source = 2
    xs = [[0, 2], [0, -2]]
    D = np.zeros((num_methods, K))

    Avg_D = np.zeros((num_source, K))

    # Receiver positions on a circle

    Phi = np.linspace(0, 2*np.pi, num=K, endpoint=False)


    #######################End of Static response######################


    # Excitation by perfet sequences.
    # p = perfect_sequence_randomphase(N)
    p = perfect_sweep(N)

    impulse_response = np.zeros((num_methods, N, K))


    #for ii in range(num_source):

    p1 = np.roll(p, int(N/2))
    distance = np.sqrt((R * np.cos(Phi) - xs[0][0]) ** 2 + (R * np.sin(Phi) - xs[0][1]) ** 2)
    delay = distance / cc
    weight = 1 / distance

    #######################Static impulse respones########################
    waveform, shift, _ = fractional_delay(delay, Lf, fs=fs, type='lagrange')
    h, _, _ = construct_ir_matrix(waveform * weight[:, np.newaxis], shift, N)
    h = h.T
    # denom = denominator(h, Phi)#  denominator of formula

    s_0, phi = initialize(Q, fs, Lf, xs[0], p)
    # print('s_0:  ')
    # print(s_0)
    s_1, _ = initialize(Q, fs, Lf, xs[1], np.roll(p, int(N / 2)))
    # print('s_1:  ')
    # print(s_1)
    s = (s_0 + s_1) / 2

    #####################################Interpolation method is linear#####################################################
    interp_method = mode
    D[0, :], impulse_response[0, :] = calc_impulse_response(K, N, s, phi, Phi, 'linear', h, p)
    D[1, :], impulse_response[1, :] = calc_impulse_response(K, N, s, phi, Phi, 'spline', h, p)
    D[2, :], impulse_response[2, :] = calc_impulse_response(K, N, s, phi, Phi, 'nearestNeighbour', h, p)
    D[3, :], impulse_response[3, :] = calc_impulse_response(K, N, s, phi, Phi, 'sinc', h, p)
    #Avg_D[0, ii] = 20*np.log10(average_fwai(D[ii, 0, :], np.linspace(90, 270, num=K)))


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
    plt.title('Impulse_Response(linear)')
    plt.imshow(impulse_response[0, :], extent=[0, 360, 0, 150], aspect="auto")
    plt.figure()
    plt.title('Impulse_Response(NearestNeighbour)')
    plt.imshow(impulse_response[1, :], extent=[0, 360, 0, 150], aspect="auto")
    plt.figure()
    plt.title('Impulse_Response(spline)')
    plt.imshow(impulse_response[2, :], extent=[0, 360, 0, 150], aspect="auto")
    plt.figure()
    plt.title('Impulse_Response(sinc)')
    plt.imshow(impulse_response[3, :], extent=[0, 360, 0, 150], aspect="auto")

    plt.figure()
    xx = np.linspace(0, 2*np.pi, num=90, endpoint=False)
    plt.xlabel(r'$\phi$/ radian')
    plt.ylabel(r'$System$ $distance$ / dB')
    plt.title('System distance')

    plt.plot(xx, db(D[0, :]), label='linear')
    plt.plot(xx, db(D[1, :]),label='spline')
    plt.plot(xx, db(D[2, :]),label='nearestNeighbour')
    plt.plot(xx, db(D[3, :]),label='sinc')
    #plt.plot(Omega_seq[0, :], y_val, label="Omega_C(Q=1.375):{}".format(Qmega_o)+"rad/s")
    plt.legend()
    plt.grid()

    #plt.xlim(0, 360)
    #plt.ylim(max_o)

    plt.show()

def retrieve_input():
    if var1.get()==True:
        inputValue = textBox.get("1.0", "end-1c")
        callback_all(float(inputValue))
    else:
        inputValue=textBox.get("1.0","end-1c")
        mode = v.get()
        callback(float(inputValue), mode)



BB=Button(top, height=1, width=10, text="Commit", command=lambda: retrieve_input())

BB.pack()
top.mainloop()




