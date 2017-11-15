import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
from utils import *

# Constants
c = 343  # speed of sound [m/s]
fs = 8000  # sampling frequency [Hz]


# Parameters
N = 150  # length of the impulse response
Omega = 2 * np.pi / 12  # angular speed of the microphone [rad/s]
K = 360  # desired number of impulse responses
Lf = 13  # length of the fractional delay filter


# Source position
xs = [0, 2]


# Receiver positions on a circle
R = 0.5  # radius
Phi = np.linspace(0, 2*np.pi, num=K, endpoint=False)
distance = np.sqrt((R*np.cos(Phi)-xs[0])**2 + (R*np.sin(Phi)-xs[1])**2)
delay = distance / c
weight = 1 / distance
waveform, shift, _ = fractional_delay(delay, Lf, fs=fs, type='lagrange')
h, _, _ = construct_ir_matrix(waveform*weight[:, np.newaxis], shift, N)


# Plot
time = np.arange(0, N) / fs * 1000

plt.figure()
plt.pcolormesh(np.rad2deg(Phi), time, h.T, cmap='coolwarm')
plt.colorbar()
plt.clim(-1, 1)
plt.xlim(0, 360)
plt.ylim(0, N/fs*1000)
plt.xlabel(r'$\varphi$ / deg')
plt.ylabel(r'$\tau$ / ms')
plt.title('Impulse Responses')
