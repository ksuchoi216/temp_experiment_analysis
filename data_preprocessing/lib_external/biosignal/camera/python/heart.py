import cv2
import time
import math
import numpy as np
import mediapipe as mp
from scipy import signal
from argparse import ArgumentParser

import cam


def convert(normalized_x, normalized_y, image_width, image_height):
    x = normalized_x * image_width
    y = normalized_y * image_height
    x_px = max(min(math.floor(x), image_width - 1), 0)
    y_px = max(min(math.floor(y), image_height - 1), 0)
    return x_px, y_px


class Queue:
    # 리스트의 사이즈를 고정하기 위한 클래스입니다.
    def __init__(self, size, value=0.0):
        self.size = size
        self.data = [value for _ in range(self.size)]

    def __len__(self):
        return len(self.data)

    def __call__(self, x):
        self.data.append(x)
        if len(self.data) > self.size:
            self.data.pop(0)

    def __getitem__(self, idx):
        return self.data[idx]


class SignalModule:
    # 심박 신호를 처리하기 위한 클래스입니다.
    def __init__(self, size=360):
        self.size = size
        self.reset()

    def reset(self):
        self.times = Queue(self.size, time.time())
        self.qr = Queue(self.size)
        self.qg = Queue(self.size)
        self.qb = Queue(self.size)
        self.qx = Queue(self.size)
        self.qy = Queue(self.size)
        self.sig = np.zeros(self.size, dtype=float)
        self.sig_filt = np.zeros(self.size, dtype=float)
        self.peaks = []
        self.frequency = 0.0

    def __call__(self, timestamp, r, g, b):
        self.times(timestamp)
        self.qr(r)
        self.qg(g)
        self.qb(b)

        # Convert to PPG signal
        x = 3*r - 2*g
        y = 1.5*r + g - 1.5*b
        self.qx(x)
        self.qy(y)

        x = np.asarray(self.qx.data)
        y = np.asarray(self.qy.data)
        beta = np.std(x) / (np.std(y) + 1e-9)
        s = x - beta*y  # PPG signal

        # TODO: check parameters
        sig = -s
        # Band pass filtering
        filt = signal.firwin(numtaps=31, cutoff=[
                             0.4, 3.0], pass_zero='bandpass', fs=15)
        sig_filt = signal.convolve(sig - sig.mean(), filt, mode='same')
        # Find peaks
        peaks, _ = signal.find_peaks(sig_filt, prominence=0.1)

        self.sig = sig
        self.sig_filt = sig_filt
        self.peaks = peaks

        # calculate frequency
        if len(peaks) >= 2:
            p0 = peaks[0]
            p1 = peaks[-1]
            t0 = self.times[p0]
            t1 = self.times[p1]
            dt = t1 - t0
            count = len(peaks)
            frequency = (count - 1) / (dt - 1e-6)
            self.frequency = frequency

    def get_graph(self, scale=160):
        # 신호 그래프 시각화 부분입니다.
        def minmax_normalize(data):
            return (data - data.min()) / (data.max() - data.min() + 1e-6)

        s = self.sig_filt
        s = (minmax_normalize(s) * scale).astype(int)
        size = len(s)

        def reorder(p0, p1):
            if p0 == p1:
                return p0, p1+1
            elif p0 > p1:
                return p1, p0
            return p0, p1

        graph1 = np.zeros((size, scale, 3), dtype=np.uint8)

        for peak in self.peaks:
            graph1[peak, :, 2] = 255
        for i in range(1, size):
            p0, p1 = reorder(s[i-1], s[i])
            graph1[i, p0:p1, 1] = 255

        v = self.sig
        v = (minmax_normalize(v) * scale).astype(int)
        graph2 = np.zeros((size, scale, 3), dtype=np.uint8)
        for i in range(1, size):
            p0, p1 = reorder(v[i-1], v[i])
            graph2[i, p0:p1, (1, 2)] = 255

        graph = np.concatenate([graph1, graph2], axis=1)

        return graph


def main(recording=False):
    cv2.namedWindow('view')
    cv2.moveWindow('view', 0, 0)
    cv2.namedWindow('hist')
    cv2.moveWindow('hist', 0, 400)
    cv2.namedWindow('crop')
    cv2.moveWindow('crop', 320, 400)
    cv2.namedWindow('mask')
    cv2.moveWindow('mask', 640, 400)

    camera = cam.Cam(0)
    img_w, img_h = 640, 360
    s_module = SignalModule(360)
    mp_face_detection = mp.solutions.face_detection

    if recording:
        rec = cam.Rec('./records', 15.0, (960, 360))

    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        while camera():
            image = camera.frame
            timestamp = camera.time

            """
            Preprocess
            이미지를 적당한 사이즈로 리사이즈합니다.
            연산량을 고려하여 사이즈를 설정합니다.
            """
            image = cv2.resize(image, (img_w, img_h))

            """
            Detect
            https://google.github.io/mediapipe/solutions/face_detection
            mediapipe를 사용하여 face detection을 합니다.
            OpenCV에서는 기본적으로 BGR 순서의 색채널을 사용하기 때문에
            RGB 채널로 변환 후 처리해야 합니다.
            """
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_detection.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image.flags.writeable = True

            """
            Convert
            검출한 bbox의 좌표를 변환합니다.
            출력된 결과는 normalized되어 있어 이미지 사이즈만큼 scaling해야합니다.
            p0 -> top left corner point
            p1 -> bottom right corner point
            """
            bboxes = []
            detections = results.detections
            if detections:
                for detection in detections:
                    location = detection.location_data
                    rel_bb = location.relative_bounding_box

                    p0 = convert(rel_bb.xmin, rel_bb.ymin, img_w, img_h)
                    p1 = convert(rel_bb.xmin+rel_bb.width,
                                 rel_bb.ymin+rel_bb.height, img_w, img_h)

                    bbox = (p0, p1)
                    bboxes.append(bbox)

            if len(bboxes):
                bbox = bboxes[0]
                # 이미지에서 bbox부분을 crop합니다.
                (x0, y0), (x1, y1) = bboxes[0]
                crop = image[y0:y1, x0:x1].copy()

                # 이미지를 HSV로 변환하고 S 채널에 대한 히스토그램을 구합니다.
                hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                hist = cv2.calcHist([hsv], [1], None, [256], [0, 256])
                hist = hist.squeeze()
                hist = signal.medfilt(hist, 5)

                # S 히스토그램에서의 최대값을 기준으로 S 값의 범위를 정합니다.
                min_pixel = 32
                hist[:min_pixel] = 0
                hist_max = np.argmax(hist)
                alpha = 0.4
                TH_range = alpha * hist_max
                range0 = int(hist_max - TH_range/2)
                range1 = int(hist_max + TH_range/2)

                sat = hsv[:, :, 1]
                mask = (sat > range0) & (sat < range1)

                # mask에 해당하는 픽셀들을 모아 R,G,B 각각의 평균값을 계산합니다.
                pixels = crop[mask]
                mean = pixels.sum(axis=0) / (len(pixels) + 1e-6)
                b, g, r = mean
                # R,G,B 평균값을 타임스탬프와 함께 signal module에 입력합니다.
                s_module(timestamp, r, g, b)

                # S 히스토그램 시각화 부분입니다.
                scale = 320
                value = np.clip(hist, 0, scale).astype(int)
                hist_view = np.zeros((256, scale, 3), np.uint8)
                for i in range(0, 256):
                    hist_view[i, :value[i], 1] = 64
                hist_view[hist_max, :, 2] = 255
                hist_view[range0, :, 0] = 255
                hist_view[range1, :, 0] = 255

                cv2.imshow('crop', crop)
                mask_view = mask.reshape(
                    mask.shape[0], mask.shape[1], 1) * crop
                cv2.imshow('mask', mask_view)
                cv2.imshow('hist', hist_view)

            # Visualize
            for p0, p1 in bboxes:
                cv2.rectangle(image, p0, p1, (0, 255, 0), 1)
            logs = [
                ' %4.1f FPS' % camera.fps,
                ' %4.1f bpm' % (s_module.frequency * 60)
            ]
            image[:50, :100] = 0
            color = (0, 255, 255)
            org = [0, 20]
            for log in logs:
                cv2.putText(image, log, org, 0, 0.5, color, 1)
                org[1] += 20
            graph = s_module.get_graph()
            view = np.concatenate([image, graph], axis=1)
            cv2.imshow('view', view)
            if recording:
                rec(view)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--recording', type=int, default=0)
    args = parser.parse_args()
    main(recording=args.recording)
