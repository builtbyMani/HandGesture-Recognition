#!/usr/bin/env python
# -*- coding: utf-8 -*-

# =========================================================
# HAND GESTURE CONTROLLED ESP32 SYSTEM
# =========================================================
#
# FEATURES:
# - MediaPipe Hand Tracking
# - Gesture Classification
# - UDP Communication to ESP32
# - Command Cooldown
# - Gesture Stabilization
# - FPS Display
# - Safe Socket Handling
#
# =========================================================

import csv
import copy
import argparse
import itertools
import socket
import time

from collections import Counter
from collections import deque

import cv2 as cv
import mediapipe as mp
import numpy as np

from utils import CvFpsCalc
from model import KeyPointClassifier
from model import PointHistoryClassifier

# =========================================================
# ESP32 CONFIGURATION
# =========================================================

ESP_IP = "192.168.1.100"   # <-- CHANGE THIS
ESP_PORT = 4210

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# =========================================================
# COMMAND SETTINGS
# =========================================================

COMMAND_MAP = {
    "Okay": "RUN",
    "Open": "STOP",
}

COMMAND_COOLDOWN = 0.5

last_command = ""
last_sent_time = 0

# =========================================================
# GESTURE STABILIZATION
# =========================================================

gesture_buffer = deque(maxlen=5)

# =========================================================
# ARGUMENTS
# =========================================================

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)

    parser.add_argument(
        '--use_static_image_mode',
        action='store_true'
    )

    parser.add_argument(
        "--min_detection_confidence",
        type=float,
        default=0.7
    )

    parser.add_argument(
        "--min_tracking_confidence",
        type=float,
        default=0.5
    )

    return parser.parse_args()

# =========================================================
# MAIN
# =========================================================

def main():

    global last_command
    global last_sent_time

    args = get_args()

    print(f"Sending UDP to {ESP_IP}:{ESP_PORT}")

    # =====================================================
    # CAMERA
    # =====================================================

    cap = cv.VideoCapture(args.device)

    cap.set(cv.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)

    # =====================================================
    # MEDIAPIPE HANDS
    # =====================================================

    mp_hands = mp.solutions.hands

    hands = mp_hands.Hands(
        static_image_mode=args.use_static_image_mode,
        max_num_hands=1,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )

    # =====================================================
    # CLASSIFIERS
    # =====================================================

    keypoint_classifier = KeyPointClassifier()
    point_history_classifier = PointHistoryClassifier()

    # =====================================================
    # LOAD LABELS
    # =====================================================

    with open(
        'model/keypoint_classifier/keypoint_classifier_label.csv',
        encoding='utf-8-sig'
    ) as f:

        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]

    with open(
        'model/point_history_classifier/point_history_classifier_label.csv',
        encoding='utf-8-sig'
    ) as f:

        point_history_classifier_labels = csv.reader(f)

        point_history_classifier_labels = [
            row[0]
            for row in point_history_classifier_labels
        ]

    # =====================================================
    # FPS
    # =====================================================

    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # =====================================================
    # HISTORY
    # =====================================================

    history_length = 16

    point_history = deque(maxlen=history_length)

    finger_gesture_history = deque(maxlen=history_length)

    mode = 0

    # =====================================================
    # MAIN LOOP
    # =====================================================

    while True:

        fps = cvFpsCalc.get()

        # =================================================
        # KEY INPUT
        # =================================================

        key = cv.waitKey(10)

        if key == 27:
            break

        number, mode = select_mode(key, mode)

        # =================================================
        # CAMERA READ
        # =================================================

        ret, image = cap.read()

        if not ret:
            break

        image = cv.flip(image, 1)

        debug_image = copy.deepcopy(image)

        # =================================================
        # MEDIAPIPE PROCESSING
        # =================================================

        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image_rgb.flags.writeable = False

        results = hands.process(image_rgb)

        image_rgb.flags.writeable = True

        # =================================================
        # HAND DETECTION
        # =================================================

        if results.multi_hand_landmarks and results.multi_handedness:

            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks,
                results.multi_handedness
            ):

                # =========================================
                # LANDMARKS
                # =========================================

                brect = calc_bounding_rect(
                    debug_image,
                    hand_landmarks
                )

                landmark_list = calc_landmark_list(
                    debug_image,
                    hand_landmarks
                )

                # =========================================
                # PREPROCESS
                # =========================================

                pre_processed_landmark_list = pre_process_landmark(
                    landmark_list
                )

                pre_processed_point_history_list = (
                    pre_process_point_history(
                        debug_image,
                        point_history
                    )
                )

                # =========================================
                # CSV LOGGING
                # =========================================

                logging_csv(
                    number,
                    mode,
                    pre_processed_landmark_list,
                    pre_processed_point_history_list
                )

                # =========================================
                # HAND SIGN CLASSIFICATION
                # =========================================

                hand_sign_id = keypoint_classifier(
                    pre_processed_landmark_list
                )

                gesture_name = (
                    keypoint_classifier_labels[hand_sign_id]
                )

                # =========================================
                # GESTURE STABILIZATION
                # =========================================

                gesture_buffer.append(gesture_name)

                stable_gesture = Counter(
                    gesture_buffer
                ).most_common(1)[0][0]

                # =========================================
                # SEND UDP COMMAND
                # =========================================

                current_time = time.time()

                if stable_gesture in COMMAND_MAP:

                    command = COMMAND_MAP[stable_gesture]

                    if (
                        command != last_command
                        or current_time - last_sent_time
                        > COMMAND_COOLDOWN
                    ):

                        try:

                            sock.sendto(
                                command.encode(),
                                (ESP_IP, ESP_PORT)
                            )

                            print(
                                f"Gesture: {stable_gesture}"
                            )

                            print(
                                f"Sent Command: {command}"
                            )

                            last_command = command
                            last_sent_time = current_time

                        except Exception as e:

                            print(
                                f"UDP Send Error: {e}"
                            )

                # =========================================
                # POINT HISTORY
                # =========================================

                if hand_sign_id == 2:
                    point_history.append(
                        landmark_list[8]
                    )
                else:
                    point_history.append([0, 0])

                # =========================================
                # FINGER GESTURE
                # =========================================

                finger_gesture_id = 0

                point_history_len = len(
                    pre_processed_point_history_list
                )

                if point_history_len == history_length * 2:

                    finger_gesture_id = (
                        point_history_classifier(
                            pre_processed_point_history_list
                        )
                    )

                finger_gesture_history.append(
                    finger_gesture_id
                )

                most_common_fg_id = Counter(
                    finger_gesture_history
                ).most_common()

                # =========================================
                # DRAWING
                # =========================================

                debug_image = draw_bounding_rect(
                    True,
                    debug_image,
                    brect
                )

                debug_image = draw_landmarks(
                    debug_image,
                    landmark_list
                )

                debug_image = draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    stable_gesture,
                    point_history_classifier_labels[
                        most_common_fg_id[0][0]
                    ],
                )

        else:

            point_history.append([0, 0])

        # =================================================
        # DRAW HISTORY + FPS
        # =================================================

        debug_image = draw_point_history(
            debug_image,
            point_history
        )

        debug_image = draw_info(
            debug_image,
            fps,
            mode,
            number
        )

        # =================================================
        # DISPLAY
        # =================================================

        cv.imshow(
            'Hand Gesture Recognition',
            debug_image
        )

        # Small delay to reduce CPU usage
        time.sleep(0.01)

    # =====================================================
    # CLEANUP
    # =====================================================

    cap.release()

    sock.close()

    cv.destroyAllWindows()

# =========================================================
# SELECT MODE
# =========================================================

def select_mode(key, mode):

    number = -1

    if 48 <= key <= 57:
        number = key - 48

    if key == 110:
        mode = 0

    if key == 107:
        mode = 1

    if key == 104:
        mode = 2

    return number, mode

# =========================================================
# BOUNDING RECTANGLE
# =========================================================

def calc_bounding_rect(image, landmarks):

    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for landmark in landmarks.landmark:

        landmark_x = min(
            int(landmark.x * image_width),
            image_width - 1
        )

        landmark_y = min(
            int(landmark.y * image_height),
            image_height - 1
        )

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(
            landmark_array,
            landmark_point,
            axis=0
        )

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]

# =========================================================
# LANDMARK LIST
# =========================================================

def calc_landmark_list(image, landmarks):

    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    for landmark in landmarks.landmark:

        landmark_x = min(
            int(landmark.x * image_width),
            image_width - 1
        )

        landmark_y = min(
            int(landmark.y * image_height),
            image_height - 1
        )

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point

# =========================================================
# PREPROCESS LANDMARKS
# =========================================================

def pre_process_landmark(landmark_list):

    temp_landmark_list = copy.deepcopy(landmark_list)

    base_x, base_y = 0, 0

    for index, landmark_point in enumerate(temp_landmark_list):

        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] -= base_x
        temp_landmark_list[index][1] -= base_y

    temp_landmark_list = list(
        itertools.chain.from_iterable(
            temp_landmark_list
        )
    )

    max_value = max(
        list(map(abs, temp_landmark_list))
    )

    if max_value == 0:
        max_value = 1

    temp_landmark_list = [
        n / max_value
        for n in temp_landmark_list
    ]

    return temp_landmark_list

# =========================================================
# PREPROCESS POINT HISTORY
# =========================================================

def pre_process_point_history(image, point_history):

    image_width, image_height = image.shape[1], image.shape[0]

    temp_point_history = copy.deepcopy(point_history)

    base_x, base_y = 0, 0

    for index, point in enumerate(temp_point_history):

        if index == 0:
            base_x, base_y = point[0], point[1]

        temp_point_history[index][0] = (
            temp_point_history[index][0] - base_x
        ) / image_width

        temp_point_history[index][1] = (
            temp_point_history[index][1] - base_y
        ) / image_height

    temp_point_history = list(
        itertools.chain.from_iterable(
            temp_point_history
        )
    )

    return temp_point_history

# =========================================================
# CSV LOGGER
# =========================================================

def logging_csv(
    number,
    mode,
    landmark_list,
    point_history_list
):

    if mode == 1 and (0 <= number <= 9):

        csv_path = (
            'model/keypoint_classifier/keypoint.csv'
        )

        with open(csv_path, 'a', newline="") as f:

            writer = csv.writer(f)

            writer.writerow(
                [number, *landmark_list]
            )

    if mode == 2 and (0 <= number <= 9):

        csv_path = (
            'model/point_history_classifier/point_history.csv'
        )

        with open(csv_path, 'a', newline="") as f:

            writer = csv.writer(f)

            writer.writerow(
                [number, *point_history_list]
            )

# =========================================================
# DRAWING FUNCTIONS
# =========================================================

def draw_landmarks(image, landmark_point):
    return image

def draw_bounding_rect(use_brect, image, brect):

    if use_brect:

        cv.rectangle(
            image,
            (brect[0], brect[1]),
            (brect[2], brect[3]),
            (0, 255, 0),
            2
        )

    return image

def draw_info_text(
    image,
    brect,
    handedness,
    hand_sign_text,
    finger_gesture_text
):

    cv.putText(
        image,
        f"{handedness.classification[0].label}: {hand_sign_text}",
        (brect[0], brect[1] - 10),
        cv.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv.LINE_AA
    )

    return image

def draw_point_history(image, point_history):

    for index, point in enumerate(point_history):

        if point[0] != 0 and point[1] != 0:

            cv.circle(
                image,
                (point[0], point[1]),
                1 + int(index / 2),
                (152, 251, 152),
                2
            )

    return image

def draw_info(image, fps, mode, number):

    cv.putText(
        image,
        f"FPS: {fps}",
        (10, 30),
        cv.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
        cv.LINE_AA
    )

    return image

# =========================================================
# START
# =========================================================

if __name__ == '__main__':
    main()