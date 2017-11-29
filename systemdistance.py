# -*- coding: utf-8 -*-


import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
from scipy import interpolate
from utils import *

def numerator(impulse_response,h):
    return sum((impulse_response-h)**2)

def denominator(h, psi):
    return sum(h**2)
    


# Constants
c = 343  # speed of sound [m/s]
fs = 8000  # sampling frequency [Hz]


# Parameters
N = 150  # length of the impulse response
Q = [8,4.57,0.5]   #12
#Omega = 2 * np.pi / Q  # angular speed of the microphone [rad/s]
K = 90  # desired number of impulse responses
Lf = 13  # length of the fractional delay filter

# Source position
xs = [0, 2]
D=np.zeros((3,K))

for ii in range(len(Q)):
    Omega = 2 * np.pi / Q[ii]  # angular speed of the microphone [rad/s]


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
denom = denominator(h, Phi)#  denominator of formula
#######################End of Static response######################


######################Dynamic impulse response##############################
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

# Excitation by perfet sequences.
#p = perfect_sequence_randomphase(N)
p = perfect_sweep(N)
# getting captured signal for each microphone
s = captured_signal(waveform, shift, p)

interp_method = 'spline'
impulse_response = np.zeros((N,K))

#for each subsignal
for k in range(K):
    y = np.zeros(N)
    for i in range(N):
        s_i = s[i::N]
        phi_i = phi[i::N] #Decompose the captured signal into N sub-signals
        y[i] = spatial_interpolation(s_i, phi_i, Phi[k], interp_method)  #interpolation

    #calculating of impulse_response
    impulse_response[:,k] = cxcorr(y, p)
##########################End of dynamic impulse response####################################################


#formula

for psi in range(K):
    nummer = numerator(impulse_response[:,psi],h[:,psi])#numerator of formula
    D[ii,psi] = 10*np.log(nummer/denom)

Phi=np.rad2deg(Phi)

# Plot
plt.figure()
plt.plot(Phi, D[0,:],label = "System distance:Omega = 8")
plt.plot(Phi, D[1,:],label = "System distance:Omega = 4.57")
plt.plot(Phi, D[2,:],label = "System distance:Omega= 0.5")
plt.legend()
plt.grid()

plt.xlim(0, 360)
plt.xlabel(r'$\varphi$ / deg')
plt.ylabel(r'$System$ $distance$ / dB')
plt.title('System Distance')
plt.show()
print("end")
