# Camera-Based Biosignal Detection in Python

## Requirements
~~~bash
pip install numpy
pip install scipy
pip install opencv-python
pip install mediapipe
~~~

<br>

## Respiratory Rate Detection
~~~bash
python3 ./resp.py

# 녹화 모드
python3 ./resp.py --recording=1
~~~

### [처리 과정](https://www.mdpi.com/1424-8220/21/15/5126)

1. RoI 선택
2. 전처리 (Resize -> Crop -> Gray)
3. Dense Optical Flow 계산
4. Y 방향 Flow 평균값 및 누적합 (Integral) 계산
5. peak detection으로 들숨/날숨 시점 도출

<br>

## Heart Rate Detection
~~~bash
python3 ./heart.py

# 녹화 모드
python3 ./heart.py --recording=1
~~~

### [처리 과정](https://www.nature.com/articles/s41598-022-11265-x)

1. Mediapipe Face Detection으로 얼굴 검출 및 해당 영역 Crop
2. HSV 변환 후 S 채널에 대한 히스토그램 생성
3. Face Skin에 해당하는 S 값의 범위 결정 및 Mask 생성
4. Mask에 해당하는 픽셀들의 R,G,B 평균값 계산
5. PPG 신호로 변환 (CHROM method)
6. Band Pass Filter -> Peak Detection

<br>

## Pose Detection
~~~bash
python3  ./pose.py

# 녹화 모드
python3 ./pose.py --recording=1
~~~

<br>

## Mediapipe
- [Holistic](https://google.github.io/mediapipe/solutions/holistic.html)
- [Face Mesh](https://google.github.io/mediapipe/solutions/face_mesh.html)
- [Pose](https://google.github.io/mediapipe/solutions/pose.html)
- [Hands](https://google.github.io/mediapipe/solutions/hands.html)

~~~bash
# Holistic
python3 mp_test.py

# Face Mesh
python3 mp_test.py --mode=1

# Pose
python3 mp_test.py --mode=2

# Hands
python3 mp_test.py --mode=3
~~~
