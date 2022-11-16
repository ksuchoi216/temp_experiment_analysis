import os
import numpy as np
from tqdm import tqdm


class SignalModel:
    def __init__(self, sampling_frequency):
        self.sampling_frequency = sampling_frequency
        self.time_step = 1 / sampling_frequency

    def __repr__(self):
        return self.__class__.__name__ + '  (sampling_frequency = {:.2f})'.format(self.sampling_frequency)

    def __call__(self, length=2048, positive=True):
        # TODO: config
        """
        f : signal frequency
        P : signal phase
        w_sigma : white noise 파라미터
        n_sigma : random walk noise 파라미터
        """
        f = np.random.uniform(0, 0.5)
        p = np.random.uniform(0, 2*np.pi)
        w_sigma = np.random.uniform(0, 1)
        n_sigma = np.random.uniform(0, 0.2)

        if positive:
            s = self.get_s(length, f, p)
        else:
            f = 0
            s = np.zeros(length)
        w = self.get_w(length, w_sigma)
        n = self.get_n(length, n_sigma)

        x = s + w + n

        x = (x - x.mean()) / (x.std() + 1e-9)  # normalize

        return {
            'f': f,
            'x': x,
            's': s,
            'w': w,
            'n': n,
        }

    def get_s(self, length, frequency, phase):
        """
        목표로 하는 생체 신호를 생성합니다.
        생체 신호 특성에 따라서 파형을 다르게 모델링할 수 있습니다.
        """
        t = np.arange(length) * self.time_step
        s = np.cos(2 * np.pi * frequency * t + phase)
        return s

    def get_w(self, length, sigma):
        """
        white noise
        """
        w = np.random.randn(length) * sigma
        return w

    def get_n(self, length, sigma):
        """
        random walk noise
        모션등에 의한 artifact를 표현한 노이즈입니다.
        """
        w = np.random.randn(length) * sigma
        n = np.cumsum(w)
        return n


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


def get_max_frequency(src, sampling_frequency, frequency_range):
    """
    frequeny_range 안에서 최대값에 해당하는 frequency를 찾는 합수입니다.
    """
    freq, value = fft(src, sampling_frequency, frequency_range)
    i_max = np.argmax(value)
    max_freq = freq[i_max]
    #max_value = value[i_max]
    return max_freq


def generate_data(
    data_path,
    data_size=1000,
    length=2048,
    sampling_frequency=32
):
    """
    모델 학습에 사용할 데이터를 생성하는 함수입니다.

    data_path : 데이터를 저장할 경로
    data_size : 생성할 데이터 수
    length : 신호 길이
    """

    """
    error_margin은 오차 허용 범위입니다.
    FFT 결과가 error_margin을 넘어가면 negative로 labeling됩니다.
    """
    frequency_range = (0.1, 1.0)
    sm = SignalModel(sampling_frequency)
    error_margin = sampling_frequency / length * 2
    print('Error Margin = ', error_margin)

    total = data_size * 2
    positives = []
    negatives = []
    for i in tqdm(range(total)):
        pack = sm(length=length, positive=True)
        sig = pack['x']
        freq = pack['f']
        pred = get_max_frequency(sig, sampling_frequency, frequency_range)
        if abs(pred - freq) < error_margin:
            positives.append(pack)
        else:
            negatives.append(pack)

    # 부족한 Negative 샘플들은 0Hz 데이터로 채워줍니다.
    remainder = data_size - len(negatives)
    for i in tqdm(range(remainder)):
        pack = sm(length=length, positive=False)
        negatives.append(pack)

    x = []
    f = []
    y = []
    for i in range(data_size):
        pack = positives[i]
        x.append(pack['x'])
        f.append(pack['f'])
        y.append(1)  # label positive

    for i in range(data_size):
        pack = negatives[i]
        x.append(pack['x'])
        f.append(pack['f'])
        y.append(0)  # label negative
    x = np.asarray(x, dtype=np.float32)
    f = np.asarray(f, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    with open(data_path, 'wb') as file:
        np.save(file, x)
        np.save(file, f)
        np.save(file, y)


def prepare_data(
    data_path='./data',
    num_train=10000,
    num_val=1000,
    num_test=2000,
):

    os.makedirs(data_path, exist_ok=True)
    data_file = os.path.join(data_path, 'train.npy')
    generate_data(data_file, data_size=num_train)
    data_file = os.path.join(data_path, 'val.npy')
    generate_data(data_file, data_size=num_val)
    data_file = os.path.join(data_path, 'test.npy')
    generate_data(data_file, data_size=num_test)


if __name__ == '__main__':
    prepare_data()
