import cv2
import numpy as np
from scipy import signal
from argparse import ArgumentParser

import cam


def preprocess(image, blur=0):
    H, W, C = image.shape
    image = cv2.resize(image, (W//2, H//2))
    image = image[H//4:]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if blur:
        # Aliasing 현상을 개선하기 위한 부분입니다. (experimental)
        gray = cv2.medianBlur(gray, blur)
        # gray = cv2.blur(gray, (blur, blur))
    return image, gray


class FlowModule:
    """
    Optical Flow를 계산하기 위한 클래스입니다.
    두 이미지가 필요하므로 이전 프레임 이미지를 prev에 기억합니다.
    https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html
    """

    def __init__(self):
        self.prev = None
        self.next = None

    def __call__(self, image):
        self.prev = self.next
        self.next = image

        if self.prev is None:
            # frame이 1번만 입력되었을 때 zeros array를 리턴합니다.
            flow = np.zeros((image.shape[0], image.shape[1], 2), np.float32)
        else:
            flow = cv2.calcOpticalFlowFarneback(
                self.prev, self.next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        return flow

    @staticmethod
    def get_flow_view(flow):
        # Optical Flow 시각화 함수입니다.
        h, w, c = flow.shape
        hsv = np.zeros((h, w, 3), dtype=np.uint8)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang*90/np.pi
        hsv[..., 1] = 255
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return bgr


class Queue:
    # 리스트의 사이즈를 고정하기 위한 클래스입니다.
    def __init__(self, size):
        self.size = size
        self.data = [0.0 for i in range(self.size)]

    def __len__(self):
        return len(self.data)

    def __call__(self, x):
        self.data.append(x)
        if len(self.data) > self.size:
            self.data.pop(0)

    def __getitem__(self, idx):
        return self.data[idx]


class SignalModule:
    # 호흡 신호를 처리하기 위한 클래스입니다.
    def __init__(self, size=360):
        self.size = size
        self.reset()

    def reset(self):
        self.times = Queue(self.size)
        self.pq = Queue(self.size)
        self.iq = Queue(self.size)
        self.integral = 0.0
        self.frequency = 0.0

    def __call__(self, timestamp, value):
        value = np.clip(value, -0.5, 0.5)
        self.times(timestamp)
        self.pq(value)
        self.integral += value
        self.iq(self.integral)

        # process
        sig = np.asarray(self.iq.data)
        # sig = signal.detrend(sig)

        # find_peaks로 들숨과 날숨 시점을 찾습니다.
        # TODO: parameter tuning
        inhalation, _ = signal.find_peaks(sig, prominence=0.05, width=5)
        exhalation, _ = signal.find_peaks(-sig, prominence=0.05, width=5)

        self.inhalation = inhalation
        self.exhalation = exhalation

        # Respiration Rate를 계산합니다. (Hz)
        if len(exhalation) >= 2:
            p0 = exhalation[0]
            p1 = exhalation[-1]
            t0 = self.times[p0]
            t1 = self.times[p1]
            dt = t1 - t0
            count = len(exhalation)
            frequency = (count - 1) / dt
            self.frequency = frequency

    def get_graph(self):
        # 신호 그래프 시각화 부분입니다.
        scale = 320
        v = (np.asarray(self.pq.data) + 0.5) * scale
        v = np.clip(v, 0, scale-1).astype(int)
        size = len(v)
        sig = np.asarray(self.iq.data)
        s = (sig - sig.min()) / (sig.max() - sig.min() + 1e-6)
        s = (s * scale).astype(int)

        graph = np.zeros((size, scale, 3), dtype=np.uint8)
        for peak in self.inhalation:
            graph[peak, :, 0] = 255
        for peak in self.exhalation:
            graph[peak, :, 2] = 255

        def reorder(p0, p1):
            if p0 == p1:
                return p0, p1+1
            elif p0 > p1:
                return p1, p0
            return p0, p1

        for i in range(1, size):
            p0, p1 = reorder(v[i-1], v[i])
            graph[i, p0:p1, 0:2] = 128
            p0, p1 = reorder(s[i-1], s[i])
            graph[i, p0:p1, 1] = 255
        return graph


def main(
    recording=False,
    blur=0,
):
    cv2.namedWindow('view')
    cv2.moveWindow('view', 0, 0)
    cv2.namedWindow('input')
    cv2.moveWindow('input', 0, 400)

    camera = cam.Cam(0)
    f_module = FlowModule()
    s_module = SignalModule()

    if recording:
        rec = cam.Rec('./records', 15.0, (960, 360))

    while camera():
        image = camera.frame
        timestamp = camera.time

        # 이미지 전처리 부분입니다.
        # Resize -> Crop -> Gray
        frame, gray = preprocess(image, blur)
        cv2.imshow('input', gray)

        # optical flow를 계산합니다.
        flow = f_module(gray)

        # y 방향의 flow의 평균값을 타임스탬프와 함께 signal module에 입력합니다.
        value = flow[..., 1].mean()
        s_module(timestamp, value)

        # Visualization
        flow_view = f_module.get_flow_view(flow)
        graph = s_module.get_graph()
        view = np.concatenate([frame, flow_view], axis=0)
        view = np.concatenate([view, graph], axis=1)

        logs = [
            ' %4.1f FPS' % camera.fps,
            ' %4.1f brpm' % (s_module.frequency * 60)
        ]
        if recording:
            logs.append(' Rec.')
        color = (0, 255, 255)
        org = [640, 40]
        for log in logs:
            cv2.putText(view, log, org, 0, 0.5, color, 1)
            org[1] += 20

        cv2.imshow('view', view)
        if recording:
            rec(view)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--recording', type=int, default=0)
    args = parser.parse_args()
    main(recording=args.recording)
