import numpy as np
from scipy import signal


def subtract_ema(src, alpha, axis=0):
    src = np.asarray(src)
    _alpha = 1 - alpha
    dst = []
    src = np.swapaxes(src, 0, axis)
    x = src[0]
    y = x
    z = x - y
    dst.append(z)
    for i in range(1, len(src)):
        x = src[i]
        y = y * _alpha + x * alpha
        z = x - y
        dst.append(z)
    dst = np.asarray(dst)
    dst = np.swapaxes(dst, 0, axis)
    return dst


def ema(src, alpha, axis=0):
    src = np.asarray(src)
    _alpha = 1 - alpha
    dst = []
    src = np.swapaxes(src, 0, axis)
    x = src[0]
    y = x
    dst.append(y)
    for i in range(1, len(src)):
        x = src[i]
        y = y * _alpha + x * alpha
        dst.append(y)
    dst = np.asarray(dst)
    dst = np.swapaxes(dst, 0, axis)
    return dst


def medfilt(src, kernel_size=None, axis=-1):
    src = np.asarray(src)
    src = np.swapaxes(src, 0, axis)
    dst = []
    for i in range(len(src)):
        dst.append(signal.medfilt(src[i], kernel_size))
    dst = np.asarray(dst)
    dst = np.swapaxes(dst, 0, axis)
    return dst


def normalize(src, axis=0):
    src = np.asarray(src)
    dst = (src - src.mean(axis, keepdims=True)) / \
        (src.std(axis, keepdims=True) + 1e-9)
    return dst


def get_diff(src, axis=0):
    src = np.asarray(src)
    src = np.swapaxes(src, 0, axis)
    prev = np.concatenate([src[0:1], src[:-1]])
    dst = src - prev
    dst = np.swapaxes(dst, 0, axis)
    return dst


def fft(src, sampling_frequency, frequency_range, axis=0):
    # return fft within given frequency range
    # src array shape: [T, M]
    # sampling_frequency: float
    # frequency_range: tuple (f0, f1) -> f0: minimum, f1: maximum
    src = np.asarray(src)
    time_step = 1.0 / sampling_frequency
    length = len(src)
    value = np.abs(np.fft.fft(src, axis=axis))
    freq = np.fft.fftfreq(length, d=time_step)
    dn = length * time_step
    i_f0 = int(frequency_range[0] * dn)
    i_f1 = int(frequency_range[1] * dn)

    value = value[i_f0:i_f1]
    freq = freq[i_f0:i_f1]
    return freq, value


def get_max_frequency(src, sampling_frequency, frequency_range):
    freq, value = np.abs(fft(src, sampling_frequency, frequency_range))
    i_max = np.argmax(value)
    return freq[i_max]


def get_max_frequency_window(timestamp, data, sampling_frequency, frequency_range, time_window=60, time_step=1):

    length = len(data)
    window = int(time_window * sampling_frequency)
    step = int(time_step * sampling_frequency)

    times = []
    freqs = []
    for i in range(window, length, step):
        crop = data[i-window:i]
        freq = get_max_frequency(crop, sampling_frequency, frequency_range)
        times.append(timestamp[i])
        freqs.append(freq)
    times = np.asarray(times)
    freqs = np.asarray(freqs)
    return times, freqs


def get_peaks(src):
    # TODO: check paramters
    peaks, _ = signal.find_peaks(
        src, prominence=75, width=[15, 250], wlen=250)
    return peaks
