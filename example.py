import numpy as np
import scipy.signal as sig
from utils import *

# Constants
c = 343

# Parameters
N = 800
Nh = int(N/2)
Q = 12
Omega = 2 * np.pi / Q
K = 360
Lf = 13
fs = 8000
L = 2 * np.pi / Omega * fs
time = np.arange(np.ceil(L)) / fs

# Excitation
p = perfect_sequence_randomphase(N)

# Receiver
R = 0.5
L = int(2 * np.pi / Omega * fs)
t = (1/fs) * np.arange(L)
phi = Omega * t
xm = [R*np.cos(phi), R*np.sin(phi)]

# Source
xs = [0, 2] # Point source
rm = np.sqrt((xm[0]-xs[0])**2 + (xm[1]-xs[1])**2)
delay = rm / c

# Captured signal
type = 'lagrange' # FD filters
waveform, shift, offset = fractional_delay(delay, Lf, fs=fs, type=type)
s = captured_signal(waveform, shift, p)

