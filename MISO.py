
import pylab as pl
from scipy import interpolate
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf, InterpolatedUnivariateSpline
import numpy as np
import scipy.signal as sig
import copy
from utils import *

def nearestneighbour(ni_i, phi_i):
    """

    """
    h = interpolate.NearestNDInterpolator(ni_i, phi_i)
    return h.x


def spatial_interpolation(s_i, phi_i, phi_target, interp_method):
    if phi_i[-1] < phi_target:
        phi_i[-1] = phi_target
    elif phi_i[0] > phi_target:
        phi_i[0] = phi_target
    s_i = np.array(s_i)
    phi_i = np.array(phi_i)
#    phi_target[k] = np.array(phi_target[k])

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





def cross_correlation(x, y):
    return cxcorr(x, y)


# Define a microphone movement (radius, angular speed, etc)
# The angular position of the microphone phi (length of L)
# Gerenate a perfect sequence p (length N)
# Compute the captured signal s, which has the same length L


# Constants
c = 343

# Parameters
N = 150 # length of the impulse reponses
Nh = int(N / 2)
Q = 8
Omega = 2 * np.pi / Q # angular speed of the microphone [rad/s]
K = 90 # desired number of impulse responses
Lf = 13 # length of the fractional delay filter
fs = 8000
L = 2 * np.pi / Omega * fs
time = np.arange(np.ceil(L)) / fs

# Source: You can change the number of sources
mic_number = 2
xs = [[0, 2],[0,-2]]  # Point source

L = int(2 * np.pi / Omega * fs)
t = (1 / fs) * np.arange(L)
#phi = Omega * t

R = 0.5 # Radius
#setting target phases
#phi_target = np.linspace(0.5*np.pi, 1.5*np.pi, num=K,endpoint= False)
phi_target = np.linspace(0, 2*np.pi, num=K,endpoint= False)



#h, _, _ = construct_ir_matrix(waveform*weight[:, np.newaxis], shift, N)

# Excitation by perfet sequences.
#p = perfect_sequence_randomphase(N)
p = perfect_sweep(N)
type = 'lagrange'  # FD filters


# initializing of spatial interplation values
y_i = np.zeros(N)


#################Select interpolation method####################################
interp_method = 'spline'
#interp_method = 'linear'
#interp_method = 'fitpack2 method'

#initializing of impulse_response
impulse_response = np.zeros((N,K))

for mic_num in range(mic_number):

    #consideration of microphone's initial phase.
    phi = Omega * t + mic_num*np.pi*2/mic_number

    # Define mic position
    distance = np.sqrt((R * np.cos(phi) - xs[0][0]) ** 2 + (R * np.sin(phi) - xs[1][1]) ** 2)

    delay = distance / c
    weight = 1 / distance
    type = 'lagrange'  # FD filters
    waveform, shift, offset = fractional_delay(delay, Lf, fs=fs, type=type)  # getting impulse_respones

    # getting captured signal for each microphone
    s = captured_signal(waveform, shift, p)

    #for each subsignal
    for k in range(K):
        y = np.zeros(N)
        for i in range(N):
            s_i = s[i::N]
            phi_i = phi[i::N]  # Decompose the captured signal into N sub-signals
            y[i] = spatial_interpolation(s_i, phi_i, phi_target[k], interp_method)  # interpolation

        # calculating of impulse_response
        impulse_response[:, k] = cxcorr(y, p)

    ##################################################################
