import numpy as np
import scipy.signal as sig
from utils import *

# Constants
c = 343                     # speed of sound [m/s]
fs = 8000                   # sampling frequency [Hz]

# Parameters
N = 800                     # excitation period
Omega = 2 * np.pi / 12      # angular speed of the microphone [rad/s]
K = 360                     # desired number of impulse responses
Lf = 13                     # length of the fractional delay filter
L = int(2*np.pi/Omega*fs)   # length of the microphone signal

# Excitation
p = perfect_sequence_randomphase(N)

# Receiver
R = 0.5                     # radius of the circular path [m]
t = (1/fs) * np.arange(L)   # time vector [s]
phi = Omega * t             # polar angule [rad]
xm = [R*np.cos(phi), R*np.sin(phi)] # microphone position [m,m]

# Point Source
xs = [0, 2]                 # position
rm = np.sqrt((xm[0]-xs[0])**2 + (xm[1]-xs[1])**2) # distance [m]
delay = rm / c              # propagation delay
weight = 1/4/np.pi/rm       # decay

# Captured signal
waveform, shift, offset = fractional_delay(delay, Lf, fs=fs, type='lagrange') # fraction delay filter
s = weight * captured_signal(waveform, shift, p) # captured signal

