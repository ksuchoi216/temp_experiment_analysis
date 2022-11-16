import cv2
import numpy as np
import mediapipe as mp
from argparse import ArgumentParser

import cam


def draw_face_landmarks(image, face_landmarks, show_coord=False):
    # https://google.github.io/mediapipe/solutions/face_mesh.html
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing.draw_landmarks(
        image,
        face_landmarks,
        mp_face_mesh.FACEMESH_TESSELATION,
        None,
        mp_drawing_styles.get_default_face_mesh_tesselation_style())
    mp_drawing.draw_landmarks(
        image,
        face_landmarks,
        mp_face_mesh.FACEMESH_CONTOURS,
        None,
        mp_drawing_styles.get_default_face_mesh_contours_style())

    if show_coord:
        color = (128, 255, 0)
        h, w, c = image.shape
        for i in [10, 152, 323, 93]:
            landmark = face_landmarks.landmark[i]
            p = int(landmark.x * w), int(landmark.y * h)
            text = ' (%.2f, %.2f, %.2f)' % (landmark.x, landmark.y, landmark.z)
            cv2.putText(image, text, p, 0, 0.5, color, 1)

    return image


def draw_pose_landmarks(image, pose_landmarks, show_coord=False):
    # https://google.github.io/mediapipe/solutions/pose.html
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose
    mp_drawing.draw_landmarks(
        image,
        pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        mp_drawing_styles.get_default_pose_landmarks_style()
    )

    if show_coord:
        color = (128, 255, 0)
        h, w, c = image.shape
        for i in [0, 7, 8, 11, 12, 13, 14, 15, 16]:
            landmark = pose_landmarks.landmark[i]
            p = int(landmark.x * w), int(landmark.y * h)
            text = ' (%.2f, %.2f, %.2f)' % (landmark.x, landmark.y, landmark.z)
            cv2.putText(image, text, p, 0, 0.5, color, 1)

    return image


def draw_hand_landmarks(image, hand_landmarks, show_coord=False):
    # https://google.github.io/mediapipe/solutions/hands.html
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands
    mp_drawing.draw_landmarks(
        image,
        hand_landmarks,
        mp_hands.HAND_CONNECTIONS,
        mp_drawing_styles.get_default_hand_landmarks_style(),
        mp_drawing_styles.get_default_hand_connections_style()
    )

    if show_coord:
        color = (128, 255, 0)
        h, w, c = image.shape
        for i in [0, 4, 8, 12, 16, 20]:
            landmark = hand_landmarks.landmark[i]
            p = int(landmark.x * w), int(landmark.y * h)
            text = ' (%.2f, %.2f, %.2f)' % (landmark.x, landmark.y, landmark.z)
            cv2.putText(image, text, p, 0, 0.5, color, 1)

    return image


def run_holistic():
    # https://google.github.io/mediapipe/solutions/holistic.html

    model = mp.solutions.holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

    camera = cam.Cam(0)
    while camera():
        image = camera.frame

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = model.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image.flags.writeable = True

        face_landmarks = results.face_landmarks
        pose_landmarks = results.pose_landmarks
        left_hand = results.left_hand_landmarks
        right_hand = results.right_hand_landmarks

        view = np.zeros(image.shape, np.uint8)
        if face_landmarks:
            view = draw_face_landmarks(view, face_landmarks, True)
        if pose_landmarks:
            view = draw_pose_landmarks(view, pose_landmarks, True)
        if left_hand:
            view = draw_hand_landmarks(view, left_hand, True)
        if right_hand:
            view = draw_hand_landmarks(view, right_hand, True)

        cv2.imshow('view', view)

    model.close()


def run_face_mesh():
    # https://google.github.io/mediapipe/solutions/face_mesh.html

    model = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

    camera = cam.Cam(0)
    while camera():
        image = camera.frame

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = model.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image.flags.writeable = True

        multi_face_landmarks = results.multi_face_landmarks

        view = np.zeros(image.shape, np.uint8)
        if multi_face_landmarks:
            for face_landmarks in multi_face_landmarks:
                view = draw_face_landmarks(view, face_landmarks, True)

        cv2.imshow('view', view)

    model.close()


def run_pose():
    # https://google.github.io/mediapipe/solutions/pose.html

    model = mp.solutions.pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=True,
        min_detection_confidence=0.5)

    camera = cam.Cam(0)
    while camera():
        image = camera.frame

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = model.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image.flags.writeable = True

        pose_landmarks = results.pose_landmarks

        view = np.zeros(image.shape, np.uint8)
        if pose_landmarks:
            view = draw_pose_landmarks(view, pose_landmarks, True)

        cv2.imshow('view', view)

    model.close()


def run_hands():
    # https://google.github.io/mediapipe/solutions/hands.html

    model = mp.solutions.hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

    camera = cam.Cam(0)
    while camera():
        image = camera.frame

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = model.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image.flags.writeable = True

        multi_hand_landmarks = results.multi_hand_landmarks

        view = np.zeros(image.shape, np.uint8)
        if multi_hand_landmarks:
            for hand_landmarks in multi_hand_landmarks:
                view = draw_hand_landmarks(view, hand_landmarks, True)

        cv2.imshow('view', view)

    model.close()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--mode', type=int, default=0)
    args = parser.parse_args()
    mode = args.mode

    if mode == 0:
        run_holistic()
    elif mode == 1:
        run_face_mesh()
    elif mode == 2:
        run_pose()
    elif mode == 3:
        run_hands()
