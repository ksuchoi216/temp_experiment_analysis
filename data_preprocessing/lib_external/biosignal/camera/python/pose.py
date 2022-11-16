import cv2
import math
import numpy as np
import mediapipe as mp
from argparse import ArgumentParser

import cam


def convert(normalized_x, normalized_y, image_width, image_height):
    x = normalized_x * image_width
    y = normalized_y * image_height
    x_px = max(min(math.floor(x), image_width - 1), 0)
    y_px = max(min(math.floor(y), image_height - 1), 0)
    return x_px, y_px


def get_points(landmarks, w, h):
    points = []
    wp = []
    visibility = []
    for landmark in landmarks.landmark:
        landmark_px = convert(landmark.x, landmark.y, w, h)
        points.append(landmark_px)
        wp.append((landmark.x-0.5, landmark.y-0.5, landmark.z))
        visibility.append(landmark.visibility)

    wp = np.asarray(wp)
    return points, wp, visibility


def main(recording=False):

    cyan = (255, 255, 0)

    status_tags = {
        0: 'Normal',
        1: 'Looking Up',
        2: 'Looking Down',
        3: 'Looking Left',
        4: 'Looking Right',
        5: 'Phone Use'
    }

    w, h = 640, 360
    if recording:
        rec = cam.Rec('./records', 15.0, (w, h))

    cv2.namedWindow('view')
    cv2.moveWindow('view', 0, 0)

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_holistic = mp.solutions.holistic
    mp_face_mesh = mp.solutions.face_mesh
    mp_hands = mp.solutions.hands

    camera = cam.Cam(0)

    with mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as holistic:
        while camera():
            image = camera.frame
            #timestamp = camera.time

            # preprocess
            image = cv2.resize(image, (w, h))
            image_r = cv2.resize(image, (w//8, h//8))

            # process
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)

            black = np.zeros_like(image)
            status = 0

            face_landmarks = results.face_landmarks
            if face_landmarks:
                mp_drawing.draw_landmarks(
                    black,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_tesselation_style())
                mp_drawing.draw_landmarks(
                    black,
                    face_landmarks,
                    mp_holistic.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_contours_style())

                points, wp, visibility = get_points(face_landmarks, w, h)

                #pT = points[10]
                #pB = points[152]
                pL = points[323]
                pR = points[93]

                wT = wp[10]
                wB = wp[152]
                wL = wp[323]
                wR = wp[93]

                wRL = wR - wL
                wBT = wB - wT
                wX = wRL / np.linalg.norm(wRL)
                wY = wBT / np.linalg.norm(wBT)
                wZ = np.cross(wX, wY)
                wZ = wZ / np.linalg.norm(wZ)

                # TODO: check
                roll = math.asin(wX[1]) / math.pi * 180
                yaw = math.asin(wZ[0]) / math.pi * 180
                pitch = math.asin(wZ[1]) / math.pi * 180

                if pitch < -10:
                    status = 1
                elif pitch > 10:
                    status = 2
                elif yaw > 25:
                    status = 3
                elif yaw < -25:
                    status = 4

                logs = [
                    'R  %+.1f' % roll,
                    'P  %+.1f' % pitch,
                    'Y  %+.1f' % yaw,
                ]

                org = [10, 100]
                for log in logs:
                    cv2.putText(black, log, org, 0, 0.5, cyan, 1)
                    org[1] += 20

            pose_landmarks = results.pose_landmarks

            if pose_landmarks:
                points, wp, visibility = get_points(pose_landmarks, w, h)

                color = (255, 255, 255)
                sl = points[11]
                sr = points[12]
                if sl is not None and sr is not None:
                    cv2.line(black, sl, sr, color, 1)

                if points[11] is not None and points[13] is not None:
                    cv2.line(black, points[11], points[13], color, 1)
                if points[12] is not None and points[14] is not None:
                    cv2.line(black, points[12], points[14], color, 1)
                if points[13] is not None and points[15] is not None:
                    cv2.line(black, points[13], points[15], color, 1)
                if points[14] is not None and points[16] is not None:
                    cv2.line(black, points[14], points[16], color, 1)

            multi_hand_landmarks = [
                results.left_hand_landmarks,
                results.right_hand_landmarks,
            ]
            for hand_landmarks in multi_hand_landmarks:
                if hand_landmarks:
                    mp_drawing.draw_landmarks(
                        black,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        None,  # mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

            if results.left_hand_landmarks:
                hand_landmarks = results.left_hand_landmarks

                points, wp, visibility = get_points(hand_landmarks, w, h)
                lhf = points[8]
                if lhf:
                    if face_landmarks:
                        dist = np.linalg.norm(np.asarray(lhf) - np.asarray(pL))
                        if dist < 48:
                            status = 5

            if results.right_hand_landmarks:
                hand_landmarks = results.right_hand_landmarks

                points, wp, visibility = get_points(hand_landmarks, w, h)
                rhf = points[8]
                if rhf:
                    if face_landmarks:
                        dist = np.linalg.norm(np.asarray(rhf) - np.asarray(pR))
                        if dist < 48:
                            status = 5

            logs = [
                ' %4.1f FPS' % camera.fps,
            ]
            if recording:
                logs.append(' Rec.')

            org = [10, 40]
            for log in logs:
                cv2.putText(black, log, org, 0, 0.5, cyan, 1)
                org[1] += 20

            log = '%s' % status_tags[status]
            org = [10, 200]
            color = (0, 128, 255) if status else (128, 255, 0)
            cv2.putText(black, log, org, 0, 0.5, color, 1)

            black[-h//8:, -w//8:] = image_r

            cv2.imshow('view', black)
            if recording:
                rec(black)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--recording', type=int, default=0)
    args = parser.parse_args()
    main(recording=args.recording)
