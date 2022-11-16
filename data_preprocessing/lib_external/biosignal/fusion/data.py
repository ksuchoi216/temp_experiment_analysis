import os
import numpy as np
from tqdm import tqdm


class SignalModel:
    def __init__(self, sampling_frequency):
        self.sampling_frequency = sampling_frequency
        self.time_step = 1 / sampling_frequency

    def get_signal(self, length, frequency, phase):
        t = np.arange(length) * self.time_step
        s = np.cos(2*np.pi*frequency*t + phase)
        return s

    def get_noise(self, length, sigma):
        return np.random.randn(length) * sigma

    def get_random_walk(self, length, sigma):
        return np.cumsum(np.random.randn(length)) * sigma

    def normalize(self, x):
        mean = x.mean()
        std = x.std()
        if std == 0:
            std = 1e-12
        x = (x - mean) / std
        return x

    def get_random_signals(self, num, length, f_range=[0.1, 0.5], p_range=[0.0, 2*np.pi], sigma_range=[0.0, 1.0], rw_sigma=0.0, e_prob=0.5):
        """
        하나의 신호로부터 서로 다른 노이즈가 추가된 센서 신호들을 생성합니다.

        num : 생성할 신호 수 (센서 수)
        length : 신호 길이
        f_range : 신호 주파수 범위
        p_range : 신호 위상 범위
        sigma_range : white noise 파라미터의 범위
        rw_sigma : random walk noise 파라미터
        e_prob : 신호 존재 확률
        """

        f = np.random.uniform(f_range[0], f_range[1])
        p = np.random.uniform(p_range[0], p_range[1])
        s = self.get_signal(length, f, p)
        s = self.normalize(s)

        xx = []
        for _ in range(num):
            e = np.random.rand() < e_prob
            sigma = np.random.uniform(sigma_range[0], sigma_range[1])
            w = self.get_noise(length, sigma)
            r = self.get_random_walk(length, rw_sigma)
            x = self.normalize(s*e + w + r)
            xx.append(x)
        xx = np.asarray(xx)
        return {
            'data': xx,
            'signal': s,
            'frequency': f,
            'phase': p}


def get_data(N, M, L, sigma_range, rw_sigma, sampling_frequency):
    """
    N : 신호 데이터 수
    M : 센서 수
    L : 신호 길이
    sigma_range : white noise 파라미터의 범위
    rw_sigma : random walk noise 파라미터
    """

    sm = SignalModel(sampling_frequency)
    x = []
    f = []
    s = []
    for _ in tqdm(range(N)):
        pack = sm.get_random_signals(
            M, L, sigma_range=sigma_range, rw_sigma=rw_sigma)
        x.append(pack['data'])
        f.append(pack['frequency'])
        s.append(pack['signal'])
    x = np.asarray(x, dtype=np.float32)
    f = np.asarray(f, dtype=np.float32)
    s = np.asarray(s, dtype=np.float32)
    return x, f, s


def generate_data(
    data_path,
    N=1000,  # Number of data
    M=16,  # Number of sensors
    L=2048,  # Signal length
    sampling_frequency=32,
):
    """
    10가지 노이즈 환경에 대한 데이터를 생성합니다.
    N : 신호 데이터 수 (N x 10 개의 데이터가 생성됩니다.)
    M : 센서 수
    L : 신호 길이
    """

    x_pack = []
    f_pack = []
    s_pack = []

    # without random walk
    for i in range(1, 6):
        sigma_range = [0, i]
        rw_sigma = 0
        x, f, s = get_data(N, M, L, sigma_range, rw_sigma, sampling_frequency)
        x_pack.append(x)
        f_pack.append(f)
        s_pack.append(s)

    # with random walk
    for i in range(1, 6):
        sigma_range = [0, 1]
        rw_sigma = i * 0.1
        x, f, s = get_data(N, M, L, sigma_range, rw_sigma, sampling_frequency)
        x_pack.append(x)
        f_pack.append(f)
        s_pack.append(s)

    x_pack = np.concatenate(x_pack, axis=0)
    f_pack = np.concatenate(f_pack, axis=0)
    s_pack = np.concatenate(s_pack, axis=0)

    with open(data_path, 'wb') as file:
        np.save(file, x_pack)
        np.save(file, f_pack)
        np.save(file, s_pack)


def prepare_data(
    data_path='./data',
    num_train=1000,
    num_val=100,
    num_test=100,
):
    """
    모델 학습에 사용할 데이터를 생성하는 함수입니다.
    data_path : 데이터를 저장할 경로
    """

    os.makedirs(data_path, exist_ok=True)
    data_file = os.path.join(data_path, 'train.npy')
    generate_data(data_file, N=num_train)
    data_file = os.path.join(data_path, 'val.npy')
    generate_data(data_file, N=num_val)
    data_file = os.path.join(data_path, 'test.npy')
    generate_data(data_file, N=num_test)


if __name__ == '__main__':
    prepare_data()
