import numpy as np
import scipy.signal as sig
from scipy import interpolate


# Perfect Sweep
def perfect_sweep(N):
    """
    generate_PerfectSweep returns a periodic perfect sweep

    Parametrs
    ---------
    N :     int
            length of the perfect sequence / sample

    Returns

    p :     array
            perfect_sweep

    """

    m = np.arange(0, np.ceil(N / 2 + 1))
    P_half = np.exp(-1j * 2 * np.pi / N * m ** 2)
    return np.real(np.fft.irfft(P_half, n=N))


def perfect_sequence_randomphase(N):
    """
    Parametrs
    ---------
    N :     int
            length of the perfect sequence / sample

    Returns

    p :     array
            perfect_sweep

    """

    m = np.arange(0, np.ceil(N / 2 + 1))
    phase = 2 * np.pi * np.random.random(len(m))
    phase[0] = 0
    P_half = np.exp(-1j * phase)
    if (N % 2) == 0:
        P_half[-1] = 1

    return np.fft.irfft(P_half, n=N)


def cconv(x, y, N=None):
    return np.fft.irfft(np.fft.rfft(x, n=N) * np.fft.rfft(y, n=N), n=N)


def cxcorr(x, y, N=None):
    return np.fft.irfft(np.fft.rfft(x) * np.fft.rfft(np.roll(y[::-1], 1)))


def time_reverse(x):
    N = len(x)
    return np.roll(x, -1)[N - 1::-1]


def db(x):
    return 20 * np.log10(np.abs(x))


def lagr_poly(ni, n):
    """Lagrange polynomail of order n

    Parameters
    ----------
    ni : array
         Sequences
    n  : scalar
         input

    Returns
    -------
    h : array
        Lagrange polynomial

    Notes
    -----

    """
    N = len(ni)
    h = np.zeros(N)
    for m in range(N):
        nm = ni[m]
        idx = np.concatenate([np.arange(0, m), np.arange(m + 1, N)])
        h[m] = np.prod((n - ni[idx]) / (nm - ni[idx]))
    return h


def fdfilt_lagr(tau, Lf, fs):
    """
    Parameters
    ----------
    tau : delay / s
    Lf : length of the filter / sample
    fs : sampling rate / Hz

    Returns
    -------
    h : (Lf)
        nonzero filter coefficients
    ni : time index of the first element of h
    n0 : time index of the center of h

    """

    d = tau * fs

    if Lf % 2 == 0:
        n0 = np.ceil(d)
        Lh = int(Lf / 2)
        idx = np.arange(n0 - Lh, n0 + Lh).astype(int)
    elif Lf % 2 == 1:
        n0 = np.round(d)
        Lh = int(np.floor(Lf / 2))
        idx = np.arange(n0 - Lh, n0 + Lh + 1).astype(int)
        idx = np.arange(n0 - Lh, n0 + Lh + 1).astype(int)
    else:
        print('Invalid value of Lf. Must be an integer')
    return lagr_poly(idx, d), idx[0], n0


def fdfilt_sinc(tau, Lf, fs, beta=8.6):
    """
    Parameters
    ----------
    tau : delay / s
    Lf : length of the filter / sample
    fs : sampling rate / Hz

    Returns
    -------
    h : (Lf)
        nonzero filter coefficients
    ni : time index of the first element of h
    n0 : time index of the center of h

    """

    d = tau * fs
    w = np.kaiser(Lf, beta)

    if Lf % 2 == 0:
        n0 = np.ceil(d)
        Lh = int(Lf / 2)
        idx = np.arange(n0 - Lh, n0 + Lh).astype(int)
    elif Lf % 2 == 1:
        n0 = np.round(d)
        Lh = int(np.floor(Lf / 2))
        idx = np.arange(n0 - Lh, n0 + Lh + 1).astype(int)
    else:
        print('Invalid value of Lf. Must be an integer')

    return np.sinc(idx - d) * w, idx[0], n0


def fdfilter(xi, yi, x, order, type='lagrange'):
    """
    Lagrange interpolation

    Parameters
    ----------
    xi :
        in accending order
    yi :

    x  :
        [xmin, xmax]

    Return
    ------
    yi :

    """
    N = order + 1
    if N % 2 == 0:
        Nhalf = N / 2
        n0 = np.searchsorted(xi, x)
        idx = np.linspace(n0 - Nhalf, n0 + Nhalf, num=N, endpoint=False).astype(int)
    elif N % 2 == 1:
        Nhalf = (N - 1) / 2
        n0 = np.argmin(np.abs(xi - x))
        idx = np.linspace(n0 - Nhalf, n0 + Nhalf + 1, num=N, endpoint=False).astype(int)
    else:
        print('order must be an integer')

    return np.dot(yi[idx], lagr_poly(xi[idx], x))


def fractional_delay(delay, Lf, fs, type):
    """
    fractional delay filter

    Parameters
    ----------
    delay : array
            time-varying delay in sample
    Lf    : int
            length of the fractional delay filter

    Returns
    -------
    waveform : array (Lf)
                nonzero coefficients
    shift    : array (Lf)
                indices of the first nonzero coefficient
    offset   : array (Lf)
                indices of the center of the filter
    """
    L = len(delay)
    waveform = np.zeros((L, Lf))
    shift = np.zeros(L)
    offset = np.zeros(L)

    if type == 'sinc':
        for n in range(L):
            htemp, ni, n0 = fdfilt_sinc(delay[n], Lf, fs=fs)
            waveform[n, :] = htemp
            shift[n] = ni
            offset[n] = n0
    elif type == 'lagrange':
        for n in range(L):
            htemp, ni, n0 = fdfilt_lagr(delay[n], Lf, fs=fs)
            waveform[n, :] = htemp
            shift[n] = ni
            offset[n] = n0
    else:
        print('unknown type')
    return waveform, shift, offset


def construct_ir_matrix(waveform, shift, Nh):
    """
    Convert 'waveform' and 'shift' into an IR matrix

    Parameters
    ----------
    waveform : array
                nonzero elements of the IRs
    shift :    array
                indices of the first nonzero coefficients
    Nh :       int
                length of each IRs

    Returns
    -------
    h :
        IRs
    H :
        TFs
    Ho :
        CHT spectrum


    """
    L, Lf = waveform.shape
    h = np.zeros((L, Nh))
    for n in range(L):
        idx = (np.arange(shift[n], shift[n] + Lf)).astype(int)
        h[n, idx] = waveform[n, :]
    H = np.fft.fft(h)
    Ho = (1 / L) * np.roll(np.fft.fft(H, axis=0), int(L / 2), axis=0)

    return h, H, Ho


def captured_signal(waveform, shift, p):
    """
    Apply time-varying delay to a perfect sweep

    Parameters
    ----------
    waveform : array
                nonzero filter coefficients
    shift :    array
                indices of the first nonzero coefficients
    p :        array
                periodic excitation signal

    Returns
    -------
    s : array
                captured signal

    """
    return time_varying_delay(waveform, shift, p)


def time_varying_delay(waveform, shift, p):
    """
    Apply a time varying delay to an input sequence
    """
    L, Lf = waveform.shape
    N = len(p)
    s = np.zeros(L)
    for n in range(L):
        idx = np.arange(shift[n], shift[n] + Lf).astype(int)
        s[n] = np.dot(p[np.mod(n - idx, N)], waveform[n, :])
    return s


def spatial_interpolation(s_i, phi_i, phi_target, interp_method):
    step_phi_i = phi_i[1] - phi_i[0]


    if phi_i[-1] < phi_target:
        delta = phi_target - phi_i[0]
        n_delta = int(np.ceil(delta / step_phi_i))
        for i in range(n_delta):
            phi_i = np.append(phi_i, phi_i[i] + 2 * np.pi)
            s_i = np.append(s_i, s_i[i])

    elif phi_i[0] > phi_target:
        delta = phi_i[0] - phi_target
        n_delta = int(np.ceil(delta / step_phi_i))
        n = len(phi_i)
        for i in range(n_delta):
            phi_i = np.insert(phi_i, 0, phi_i[n - 1 - i] - 2 * np.pi)
            s_i = np.insert(s_i, 0, s_i[n - 1 - i])

    """
    if np.amax(phi_i) < phi_target:
        delta = phi_target - phi_i[0]
        n_delta = int(np.ceil(delta / step_phi_i))
        for i in range(n_delta):
            phi_i = np.append(phi_i, phi_i[i] + 2 * np.pi)
            s_i = np.append(s_i, s_i[i])

    elif np.amin(phi_i) > phi_target:
        delta = phi_i[0] - phi_target
        n_delta = int(np.ceil(delta / step_phi_i))
        n = len(phi_i)
        for i in range(n_delta):
            phi_i = np.insert(phi_i, 0, phi_i[n - 1 - i] - 2 * np.pi)
            s_i = np.insert(s_i, 0, s_i[n - 1 - i])
    """
    if interp_method == 'linear':
        return linear(s_i, phi_i, phi_target)
        #return
        #f = interpolate.interp1d(phi_i, s_i, kind='linear', bounds_error=False)
        #h = f(phi_target)
    elif interp_method == 'sinc':
        return sinc_interp(phi_i, s_i, phi_target)

    elif interp_method == 'nearestNeighbour':
        return NN(s_i, phi_i, phi_target)
        #ius = interpolate.interp1d(phi_i, s_i, kind='nearest')
        #h = ius(phi_target)
    elif interp_method == 'spline':
        return spline(s_i, phi_i, phi_target)
    else:
        print("Please select correct interpolation method")
        return



def numerator(impulse_response, h):
    return sum((impulse_response - h) ** 2)


def denominator(h):
    return sum(h ** 2)


def sinc_interp(s_i, phi_i, phi_target):
    if len(phi_i) != len(s_i):
        raise ValueError('x and s must be the same length')

    # Find the period
    T = s_i[1] - s_i[0]

    sincM = np.tile(phi_target, (np.size(s_i), 1)) - np.tile(s_i[:, np.newaxis], (1, np.size(phi_target)))
    y = np.dot(phi_i, np.sinc(sincM / T))
    return y

"""
def average_fwai(h):
    #val = 0
    #sum_psai = np.sum(psai)
    #sum_square_h = sum((h ** 2) * psai)
    #val = sum_square_h/sum_psai
    return np.mean(h)#val
"""

def NN(s_i, phi_i, phi_target):
<<<<<<< HEAD
    tck= interpolate.interp1d(phi_i, s_i, kind='nearest', bounds_error=False)
    return tck(phi_target)
=======
    ius = interpolate.interp1d(phi_i, s_i, kind='nearest', bounds_error=False)
    return ius(phi_target)

>>>>>>> 05ffd622472e205501dc1f531baddc8a26102594

def linear(s_i, phi_i, phi_target):
    f = interpolate.interp1d(phi_i, s_i, kind='linear', bounds_error=False)
    return f(phi_target)

def spline(s_i, phi_i, phi_target):
    f = interpolate.interp1d(phi_i, s_i, kind='slinear', bounds_error=False)
    return f(phi_target)
