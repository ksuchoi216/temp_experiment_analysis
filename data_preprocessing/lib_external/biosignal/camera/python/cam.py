import os
import cv2
import time
from datetime import datetime


class Cam:
    def __init__(self, index=0):
        self.cap = cv2.VideoCapture(index)
        self.W = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.H = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # for FPS
        self.fps = 0.0
        self.count = 0
        self.time_prev = time.time()

    def __call__(self):
        success, frame = self.cap.read()
        self.time = time.time()
        self.frame = frame

        # for FPS
        self.count += 1
        check = 30
        if self.count % check == 0:
            self.fps = check / (self.time - self.time_prev)
            self.time_prev = self.time

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            return False
        return success

    def __del__(self):
        self.cap.release()

    def release(self):
        print('cap. release')
        self.cap.release()


class Rec:
    def __init__(self, path, fps, size):
        os.makedirs(path, exist_ok=True)
        now = datetime.now()
        now = now.strftime("%Y%m%d-%H%M%S")
        rec_name = os.path.join(path, '%s.mp4' % now)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(rec_name, fourcc, fps, size)

    def __call__(self, image):
        self.write(image)

    def __del__(self):
        self.release()

    def write(self, image):
        self.writer.write(image)

    def release(self):
        print('rec. release')
        self.writer.release()


def test():
    camera = Cam()
    while camera():
        image = cv2.resize(camera.frame, (320, 180))
        cv2.imshow('frame', image)
        print(camera.fps)


if __name__ == '__main__':
    test()
