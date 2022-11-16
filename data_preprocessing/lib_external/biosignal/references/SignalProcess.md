# Signal Process

신호 처리에 자주 사용되는 함수들입니다.

---

<br>

## FIR (Finite Impulse Response) filter

- FIR 필터는 [signal.firwin](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.firwin.html)과 [signal.convolve](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.convolve.html)로 적용할 수 있습니다.


~~~python
from scipy import signal

filt = signal.firwin(
    numtaps=256, cutoff=[1.0, 5.0], pass_zero='bandpass', fs=64)
y = signal.convolve(x, filt, mode='same')
~~~


## Normalization

~~~python
def normalize(src, axis=0)
    mean = src.mean(axis, keepdims=True)
    std = src.std(axis, keepdims=True)
    if std == 0.0:
        std = 1e-9
    dst = (src - mean) / std

y = normalize(x)
~~~

## Find Peaks

- [signal.find_peaks](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html)

~~~python
from scipy import signal

y = normalize(x)
peaks, _ = signal.find_peaks(y, prominence=0.5)
~~~


## FFT (Fast Fourier Transform)
- [np.fft.fft](https://numpy.org/doc/stable/reference/generated/numpy.fft.fft.html)
- [np.fft.fftfreq](https://numpy.org/doc/stable/reference/generated/numpy.fft.fftfreq.html)

~~~python
import numpy as np
import matplotlib.pyplot as plt

def fft(src, sampling_frequency, frequency_range, axis=0):
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

freq, value = fft(x, 64, [1, 5])
plt.plot(freq, value)
~~~

