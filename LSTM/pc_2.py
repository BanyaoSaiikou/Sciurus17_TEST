# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 18:18:40 2020

@author: tuan
"""

import sys
sys.path.append('../')
import os
import math
import scipy.stats as ss
import numpy as np
import copy
import cv2 as cv
import socket
import pickle
import util
import matplotlib.pyplot as plt
from random import randint
from scipy.ndimage.filters import gaussian_filter
from model import get_testing_model
from py_rmpe_server.py_rmpe_config import RmpeGlobalConfig
from training.lstm import *
from testing.config_reader import config_reader
from dataset.definition import *

WEIGHT_FILE_IDX = "0006"

TCP_IP = "10.40.1.84"
#TCP_IP = "133.19.43.1"#CAI Weihao 改这里
TCP_PORT = 5432
BUFFER_SIZE = 1024

BOXES = ["blue", "yellow", "green"]
COLORS = [(0, 0, 255), (255, 0, 255), (0, 255, 255)]

OFFSET_Y = 204  # Y and X is different form robot code
OFFSET_X = 32
SCALE = 1000
SCALE_F = 1000.0
N_BOXES = len(BOXES)
BOX_W = 60
Y_MIN = 0 + OFFSET_Y
Y_MAX = 100 + OFFSET_Y
X_MIN = -180 + OFFSET_X
X_MAX = 220 + OFFSET_X
DIST_MIN = int(BOX_W * math.sqrt(2)) + 20  # 50
DIST_MAX = 20
Z_0 = 50
N_STATES = 46
NONE_ACTION_IDX = 24

RAND_POSITIONS = [[-111, 420], [6, 420], [123, 420],#关键位置
                  [-111, 240], [6, 240], [123, 240]]
ZERO_POSITIONS = [[0, 0, 0, 0] for i in range(3)]

out_img_idx = 0
curr_positions = copy.deepcopy(ZERO_POSITIONS)


def test_socket():#创建套接字并链接至远端地址
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((TCP_IP, TCP_PORT))

    while 1:
        message = input("Message : ")
        s.send(message.encode())
        data = pickle.loads(s.recv(BUFFER_SIZE))#接收数据
        print("Received data:", data)

        is_continue = int(input("Continue? : "))
        if is_continue == 0:
            break

    s.close()


def rand_valid_position(box_idx, first_rand):
    if first_rand:
        return 35, 275, Z_0

    for pos_i in range(len(RAND_POSITIONS)):
        x = RAND_POSITIONS[pos_i][0]
        y = RAND_POSITIONS[pos_i][1]
        is_valid = True
        for i in range(len(BOXES)):
            if i == box_idx:
                continue
            dist = math.sqrt((curr_positions[i][0] - x)**2 +
                             (curr_positions[i][1] - y)**2)
            if dist <= DIST_MIN:
                is_valid = False
        if is_valid:
            break

    return x, y, Z_0


def rand_valid_position_2(box_idx):
    return 35, 275, Z_0
    while True:
        x = randint(X_MIN, X_MAX)
        y = randint(Y_MIN, Y_MAX)
        is_valid = True
        for i in range(len(BOXES)):
            if i == box_idx:
                continue
            dist = math.sqrt((curr_positions[i][0] - x)**2 +
                             (curr_positions[i][1] - y)**2)
            if dist <= DIST_MIN:
                is_valid = False
        if is_valid:
            break

    return x, y, Z_0


def change_coordinate_system():

    for i in range(len(BOXES)):
        if curr_positions[i][3] == 0:
            curr_positions[i][3] = 1


def action_to_position(action_idx, is_first_move, is_first_rand):

    box_idx = -1

    if is_first_move:
        change_coordinate_system()

    next_positions = copy.deepcopy(curr_positions)

    if action_idx == 0 or action_idx == 8 or action_idx == 16:
        if action_idx == 0:
            box_idx = 0
        elif action_idx == 8:
            box_idx = 1
        elif action_idx == 16:
            box_idx = 2

        x, y, z = rand_valid_position(box_idx, is_first_rand)
        if is_first_rand:
            is_first_rand = False

        next_positions[box_idx] = [x, y, z, 1]

    elif action_idx >= 1 and action_idx <= 7:
        box_idx = 0
        if action_idx == 1:
            next_positions[box_idx] = [next_positions[1][0] - BOX_W,
                                       next_positions[1][1], Z_0, 1]
        elif action_idx == 2:
            next_positions[box_idx] = [next_positions[1][0] + BOX_W,
                                       next_positions[1][1], Z_0, 1]
        elif action_idx == 3:
            next_positions[box_idx] = [next_positions[2][0] - BOX_W,
                                       next_positions[2][1], Z_0, 1]
        elif action_idx == 4:
            next_positions[box_idx] = [next_positions[2][0] + BOX_W,
                                       next_positions[2][1], Z_0, 1]
        elif action_idx == 5:
            next_positions[box_idx] = [next_positions[1][0],
                                       next_positions[1][1],
                                       next_positions[1][2] + BOX_W, 1]
        elif action_idx == 6:
            next_positions[box_idx] = [next_positions[2][0],
                                       next_positions[2][1],
                                       next_positions[2][2] + BOX_W, 1]
        elif action_idx == 7:
            next_positions[box_idx] = [int((next_positions[1][0] +
                                            next_positions[2][0])/2),
                                       int((next_positions[1][1] +
                                            next_positions[2][1])/2),
                                       BOX_W, 1]

    elif action_idx >= 9 and action_idx <= 15:
        box_idx = 1
        if action_idx == 9:
            next_positions[box_idx] = [next_positions[0][0] - BOX_W,
                                       next_positions[0][1], Z_0, 1]
        elif action_idx == 10:
            next_positions[box_idx] = [next_positions[0][0] + BOX_W,
                                       next_positions[0][1], Z_0, 1]
        elif action_idx == 11:
            next_positions[box_idx] = [next_positions[2][0] - BOX_W,
                                       next_positions[2][1], Z_0, 1]
        elif action_idx == 12:
            next_positions[box_idx] = [next_positions[2][0] + BOX_W,
                                       next_positions[2][1], Z_0, 1]
        elif action_idx == 13:
            next_positions[box_idx] = [next_positions[0][0],
                                       next_positions[0][1],
                                       next_positions[0][2] + BOX_W, 1]
        elif action_idx == 14:
            next_positions[box_idx] = [next_positions[2][0],
                                       next_positions[2][1],
                                       next_positions[2][2] + BOX_W, 1]
        elif action_idx == 15:
            next_positions[box_idx] = [int((next_positions[0][0] +
                                            next_positions[2][0])/2),
                                       int((next_positions[0][1] +
                                            next_positions[2][1])/2),
                                       BOX_W, 1]

    elif action_idx >= 17 and action_idx <= 23:
        box_idx = 2
        if action_idx == 17:
            next_positions[box_idx] = [next_positions[0][0] - BOX_W,
                                       next_positions[0][1], Z_0, 1]
        elif action_idx == 18:
            next_positions[box_idx] = [next_positions[0][0] + BOX_W,
                                       next_positions[0][1], Z_0, 1]
        elif action_idx == 19:
            next_positions[box_idx] = [next_positions[1][0] - BOX_W,
                                       next_positions[1][1], Z_0, 1]
        elif action_idx == 20:
            next_positions[box_idx] = [next_positions[1][0] + BOX_W,
                                       next_positions[1][1], Z_0, 1]
        elif action_idx == 21:
            next_positions[box_idx] = [next_positions[0][0],
                                       next_positions[0][1],
                                       next_positions[0][2] + BOX_W, 1]
        elif action_idx == 22:
            next_positions[box_idx] = [next_positions[1][0],
                                       next_positions[1][1],
                                       next_positions[1][2] + BOX_W, 1]
        elif action_idx == 23:
            next_positions[box_idx] = [int((next_positions[0][0]
                                            + next_positions[1][0])/2),
                                       int((next_positions[0][1] +
                                            next_positions[1][1])/2),
                                       BOX_W, 1]

    return box_idx, next_positions, is_first_rand


def position_to_state(action_idx, is_first_move, first_state):
    if is_first_move:
        if action_idx == 0:
            return 1
        elif action_idx == 8:
            return 2
        elif action_idx == 16:
            return 3

    # Blue and yellow
    pos_0_1 = [curr_positions[0][0] - curr_positions[1][0],
               curr_positions[0][1] - curr_positions[1][1],
               curr_positions[0][2] - curr_positions[1][2],
               curr_positions[0][3] - curr_positions[1][3]]
    state_0_1 = 0
    if abs(pos_0_1[3]) == 1 or (curr_positions[0][3] == 0 and
                                curr_positions[1][3] == 0):
        state_0_1 = 0
    elif abs(pos_0_1[1]) > DIST_MAX:
        state_0_1 = 0
    elif abs(pos_0_1[0]) > DIST_MAX + BOX_W:
        state_0_1 = 0
    elif abs(pos_0_1[2]) > DIST_MAX + BOX_W:
        state_0_1 = 0
    elif (abs(pos_0_1[2]) <= DIST_MAX and
          (abs(pos_0_1[0]) >= BOX_W - DIST_MAX or
           abs(pos_0_1[1]) >= BOX_W - DIST_MAX)):
        if pos_0_1[0] < 0:
            state_0_1 = 9
        elif pos_0_1[0] > 0:
            state_0_1 = 11
    elif ((abs(pos_0_1[2]) >= BOX_W - DIST_MAX) and
          (abs(pos_0_1[2]) <= BOX_W + DIST_MAX) and
          (abs(pos_0_1[0]) <= DIST_MAX)):
        if pos_0_1[2] < 0:
            state_0_1 = 25
        elif pos_0_1[2] > 0:
            state_0_1 = 27
    elif ((abs(pos_0_1[2]) >= BOX_W - DIST_MAX) and
          (abs(pos_0_1[0]) <= BOX_W/2 + DIST_MAX)):
        state_0_1 = -1

    # Blue and green
    pos_0_2 = [curr_positions[0][0] - curr_positions[2][0],
               curr_positions[0][1] - curr_positions[2][1],
               curr_positions[0][2] - curr_positions[2][2],
               curr_positions[0][3] - curr_positions[2][3]]
    state_0_2 = 0
    if abs(pos_0_2[3]) == 1 or (curr_positions[0][3] == 0
                                and curr_positions[1][3] == 0):
        state_0_2 = 0
    elif abs(pos_0_2[1]) > DIST_MAX:
        state_0_2 = 0
    elif abs(pos_0_2[0]) > DIST_MAX + BOX_W:
        state_0_2 = 0
    elif abs(pos_0_2[2]) > DIST_MAX + BOX_W:
        state_0_2 = 0
    elif (abs(pos_0_2[2]) <= DIST_MAX and
          (abs(pos_0_2[0]) >= BOX_W - DIST_MAX or
           abs(pos_0_2[1]) >= BOX_W - DIST_MAX)):
        if pos_0_2[0] < 0:
            state_0_2 = 10
        elif pos_0_2[0] > 0:
            state_0_2 = 18
    elif ((abs(pos_0_2[2]) >= BOX_W - DIST_MAX) and
          (abs(pos_0_2[2]) <= BOX_W + DIST_MAX) and
          (abs(pos_0_2[0]) <= DIST_MAX)):
        if pos_0_2[2] < 0:
            state_0_2 = 26
        elif pos_0_2[2] > 0:
            state_0_2 = 35
    elif ((abs(pos_0_2[2]) >= BOX_W - DIST_MAX) and
          (abs(pos_0_2[0]) <= BOX_W/2 + DIST_MAX)):
        state_0_2 = -1

    # Yellow and green
    pos_1_2 = [curr_positions[1][0] - curr_positions[2][0],
               curr_positions[1][1] - curr_positions[2][1],
               curr_positions[1][2] - curr_positions[2][2],
               curr_positions[1][3] - curr_positions[2][3]]
    state_1_2 = 0
    if abs(pos_1_2[3]) == 1 or (curr_positions[0][3] == 0 and
                                curr_positions[1][3] == 0):
        state_1_2 = 0
    elif abs(pos_1_2[1]) > DIST_MAX:
        state_1_2 = 0
    elif abs(pos_1_2[0]) > DIST_MAX + BOX_W:
        state_1_2 = 0
    elif abs(pos_1_2[2]) > DIST_MAX + BOX_W:
        state_1_2 = 0
    elif (abs(pos_1_2[2]) <= DIST_MAX and
          (abs(pos_1_2[0]) >= BOX_W - DIST_MAX or
           abs(pos_1_2[1]) >= BOX_W - DIST_MAX)):
        if pos_1_2[0] < 0:
            state_1_2 = 17
        elif pos_1_2[0] > 0:
            state_1_2 = 19
    elif ((abs(pos_1_2[2]) >= BOX_W - DIST_MAX) and
          (abs(pos_1_2[2]) <= BOX_W + DIST_MAX) and
          (abs(pos_1_2[0]) <= DIST_MAX)):
        if pos_1_2[2] < 0:
            state_1_2 = 34
        elif pos_1_2[2] > 0:
            state_1_2 = 36
    elif ((abs(pos_1_2[2]) >= BOX_W - DIST_MAX) and
          (abs(pos_1_2[0]) <= BOX_W/2 + DIST_MAX)):
        state_1_2 = -1

    # print(state_0_1)
    # print(state_0_2)
    # print(state_1_2)

    if state_0_1 == 0 and state_0_2 == 0 and state_1_2 == 0:
        return first_state

    if state_0_1 != 0 and state_0_2 != 0 and state_1_2 != 0:
        dist_0_1 = pos_0_1[0]*pos_0_1[0] + pos_0_1[1]*pos_0_1[1] + \
            pos_0_1[2]*pos_0_1[2]
        dist_0_2 = pos_0_2[0]*pos_0_2[0] + pos_0_2[1]*pos_0_2[1] + \
            pos_0_2[2]*pos_0_2[2]
        dist_1_2 = pos_1_2[0]*pos_1_2[0] + pos_1_2[1]*pos_1_2[1] + \
            pos_1_2[2]*pos_1_2[2]
        if dist_0_1 >= dist_0_2:
            if dist_0_1 >= dist_1_2:
                state_0_1 = 0
            else:
                state_1_2 = 0
        else:
            if dist_0_2 >= dist_1_2:
                state_0_2 = 0
            else:
                state_1_2 = 0

    # Blue and yellow
    if state_0_1 == 9:
        if state_0_2 == 0 and state_1_2 == 0:
            return state_0_1
        if state_0_2 == 18 and state_1_2 == 0:
            return 20
        if state_0_2 == 0 and state_1_2 == 17:
            return 4
        if state_0_2 == 26 and state_1_2 == 0:
            return 24
        if state_0_2 == 0 and state_1_2 == 34:
            return 42
        if state_0_2 == -1 and state_1_2 == -1:
            return 8

    if state_0_1 == 11:
        if state_0_2 == 0 and state_1_2 == 0:
            return state_0_1
        if state_0_2 == 10 and state_1_2 == 0:
            return 12
        if state_0_2 == 0 and state_1_2 == 19:
            return 21
        if state_0_2 == 26 and state_1_2 == 0:
            return 45
        if state_0_2 == 0 and state_1_2 == 34:
            return 33
        if state_0_2 == -1 and state_1_2 == -1:
            return 16

    if state_0_1 == 25:
        if state_0_2 == 0 and state_1_2 == 0:
            return state_0_1
        if state_0_2 == 10 and state_1_2 == 0:
            return 32
        if state_0_2 == 18 and state_1_2 == 0:
            return 41
        if state_0_2 == 0 and state_1_2 == 34:
            return 28

    if state_0_1 == 27:
        if state_0_2 == 0 and state_1_2 == 0:
            return state_0_1
        if state_0_2 == 0 and state_1_2 == 17:
            return 22
        if state_0_2 == 0 and state_1_2 == 19:
            return 43
        if state_0_2 == 26 and state_1_2 == 0:
            return 30

    # Blue and green
    if state_0_2 == 10:
        if state_0_1 == 0 and state_1_2 == 0:
            return state_0_2
        if state_0_1 == 0 and state_1_2 == 19:
            return 5
        if state_0_1 == 0 and state_1_2 == 36:
            return 44
        if state_0_1 == -1 and state_1_2 == -1:
            return 15

    if state_0_2 == 18:
        if state_0_1 == 0 and state_1_2 == 0:
            return state_0_2
        if state_0_1 == 0 and state_1_2 == 17:
            return 13
        if state_0_1 == 0 and state_1_2 == 36:
            return 23
        if state_0_1 == -1 and state_1_2 == -1:
            return 7

    if state_0_2 == 26:
        if state_0_1 == 0 and state_1_2 == 0:
            return state_0_2
        if state_0_1 == 0 and state_1_2 == 36:
            return 29

    if state_0_2 == 35:
        if state_0_1 == 0 and state_1_2 == 0:
            return state_0_2
        if state_0_1 == 0 and state_1_2 == 17:
            return 40
        if state_0_1 == 0 and state_1_2 == 19:
            return 31
        if state_0_1 == 25 and state_1_2 == 0:
            return 38

    # Yellow and green
    if state_1_2 == 17:
        if state_0_1 == 0 and state_0_2 == 0:
            return state_1_2
        if state_0_1 == -1 and state_0_2 == -1:
            return 6

    if state_1_2 == 19:
        if state_0_1 == 0 and state_0_2 == 0:
            return state_1_2
        if state_0_1 == -1 and state_0_2 == -1:
            return 14

    if state_1_2 == 34:
        if state_0_1 == 0 and state_0_2 == 0:
            return state_1_2
        if state_0_1 == 0 and state_0_2 == 35:
            return 37

    if state_1_2 == 36:
        if state_0_1 == 0 and state_0_2 == 0:
            return state_1_2
        if state_0_1 == 27 and state_0_2 == 0:
            return 39

    # More
    # if state_0_1 == 25 and pos_1_2[2] < 0:
    #     return 28
    # if state_0_1 == 27 and pos_0_2[2] < 0:
    #     return 30

    # if state_0_2 == 26 and pos_1_2[2] > 0:
    #     return 29
    # if state_0_2 == 35 and pos_0_1[2] < 0:
    #     return 38

    # if state_1_2 == 34 and pos_0_2[2] > 0:
    #     return 37
    # if state_1_2 == 36 and pos_0_1[2] > 0:
    #     return 39

    return 0


def draw_curr_state(state_idx):
    print("Current state detected by CNN:", state_idx)
    fig = plt.figure(figsize=(1, 0.5))
    fig.add_subplot(1, 1, 1)
    state_img = cv.imread("../dataset/state_images/" + str(state_idx) + ".png")
    state_img = cv.cvtColor(state_img, cv.COLOR_BGR2RGB)
    plt.axis("off")
    plt.imshow(state_img)
    plt.show()


def estimate_keypoints_child(models, input_img_0):
    output_img = input_img_0.copy()
    all_box_peaks = []

    for model_idx in range(len(models)):
        model = models[model_idx]

        input_img = input_img_0.copy()

        params, model_params = config_reader()

        multiplier = [x * model_params["boxsize"] / input_img.shape[0]
                      for x in params["scale_search"]]
        heatmap_avg = np.zeros((input_img.shape[0], input_img.shape[1],
                                RmpeGlobalConfig.num_parts_with_background))

        for m in range(len(multiplier)):

            scale = multiplier[m]
            img_to_test = cv.resize(input_img, (0, 0), fx=scale, fy=scale,
                                    interpolation=cv.INTER_CUBIC)
            img_padded, pad = util.padRightDownCorner(img_to_test,
                                                      model_params["stride"],
                                                      model_params["padValue"])

            trans_in_img = np.transpose(
                np.float32(img_padded[:, :, :, np.newaxis]),
                (3, 0, 1, 2))  # Required shape (1, width, height, channels)

            output_blobs = model.predict(trans_in_img)

            # Extract outputs, resize, and remove padding
            heatmap = np.squeeze(output_blobs[0])  # Output 1 is heatmaps
            heatmap = cv.resize(heatmap, (0, 0), fx=model_params["stride"],
                                fy=model_params["stride"],
                                interpolation=cv.INTER_CUBIC)
            heatmap = heatmap[:img_padded.shape[0] - pad[2],
                              :img_padded.shape[1] - pad[3], :]
            heatmap = cv.resize(heatmap,
                                (input_img.shape[1], input_img.shape[0]),
                                interpolation=cv.INTER_CUBIC)

            # heatmap_avg = heatmap_avg + heatmap
            heatmap_avg = np.maximum(heatmap_avg, heatmap)

        all_peaks = []
        all_peaks_with_score_and_id = []
        peak_counter = 0

        for part in range(RmpeGlobalConfig.num_parts):
            map_ori = heatmap_avg[:, :, part]
            map = gaussian_filter(map_ori, sigma=3)

            map_left = np.zeros(map.shape)
            map_left[1:, :] = map[:-1, :]
            map_right = np.zeros(map.shape)
            map_right[:-1, :] = map[1:, :]
            map_up = np.zeros(map.shape)
            map_up[:, 1:] = map[:, :-1]
            map_down = np.zeros(map.shape)
            map_down[:, :-1] = map[:, 1:]

            peaks_binary = np.logical_and.reduce(
                (map >= map_left, map >= map_right,
                 map >= map_up, map >= map_down, map > params["thre1"]))
            peaks = list(zip(np.nonzero(peaks_binary)[1],
                             np.nonzero(peaks_binary)[0]))  # Note reverse

            if(len(peaks) > 0):
                max_score = 0
                max_peaks = peaks[0]
                for x in peaks:
                    if map_ori[x[1], x[0]] >= max_score:
                        max_score = map_ori[x[1], x[0]]
                        max_peaks = x

                peaks = list([max_peaks])

            all_peaks.append(peaks)
            peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]
            id = range(peak_counter, peak_counter + len(peaks))
            peaks_with_score_and_id = [peaks_with_score[i] + (id[i],)
                                       for i in range(len(id))]

            all_peaks_with_score_and_id.append(peaks_with_score_and_id)
            peak_counter += len(peaks)

        # print(all_peaks)
        all_box_peaks.append(all_peaks)

        for i, peaks in enumerate(all_peaks):
            if(len(peaks) > 0):
                if i == 0:
                    output_img = cv.circle(output_img, peaks[0], 3,
                                           COLORS[0], -1)
                elif i == 1:
                    output_img = cv.circle(output_img, peaks[0], 3,
                                           COLORS[1], -1)
                # else:
                #     output_img = cv.circle(output_img, peaks[0], 2,
                #                            COLORS[2], -1)

    return output_img, all_box_peaks


def load_cnn_models():
    models = []
    for i in range(len(RmpeGlobalConfig.box_types)):
        weight_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..",
                         "training",
                         "weights_" + RmpeGlobalConfig.box_types[i][1] + "_" +
                         str(RmpeGlobalConfig.num_parts)))
        keras_weights_file = os.path.join(weight_dir, "weights." +
                                          WEIGHT_FILE_IDX + ".h5")

        model = get_testing_model()
        model.load_weights(keras_weights_file)
        models.append(model)

    return models


def estimate_keypoints(models, input_img):
    global out_img_idx

    canvas, all_box_peaks = estimate_keypoints_child(models, input_img)
    cv.imwrite(str(out_img_idx) + ".png", canvas)
    out_img_idx += 1

    # output_folder = "output"
    # output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),
    #                                           output_folder))
    # os.makedirs(output_dir, exist_ok=True)
    # output_image_dir = os.path.join(output_dir, file_name + ext)
    # cv.imwrite(output_image_dir, canvas)

    keypoints = []
    for i in range(len(all_box_peaks)):
        keypoints.append([])
        for j in range(2):
            keypoint = all_box_peaks[i][j]
            if keypoint != []:
                keypoint = keypoint[0]
                keypoints[i].append([keypoint[0], keypoint[1]])
            else:
                keypoints[i].append([0, 0])

    # print(keypoints)
    return keypoints


def recvall(sock, count):
    buf = b""
    while count:
        newbuf = sock.recv(count)
        if not newbuf:
            return None
        buf += newbuf
        count -= len(newbuf)
    return buf


def recv_imgs_and_find_keypoints(models):
    while True:
        checker = pickle.loads(s.recv(BUFFER_SIZE))
        # print(checker)
        if checker[0] == 0:
            break
        length = recvall(s, 16)
        stringData = recvall(s, int(length))
        data = np.frombuffer(stringData, dtype="uint8")
        decimg = cv.imdecode(data, 1)
        keypoints = estimate_keypoints(models, decimg)
        s.send(pickle.dumps([1, keypoints], protocol=2))

    return


def assign_position(data):
    for i in range(len(BOXES)):
        if data[i][0][0] != 0 and data[i][0][1] != 0 and data[i][0][2] != 0:
            curr_positions[i] = [data[i][0][1]*SCALE,
                                 data[i][0][0]*SCALE,
                                 data[i][0][2]*SCALE,
                                 curr_positions[i][3]]


def assign_position_2(data):
    side_positions = copy.deepcopy(ZERO_POSITIONS)
    for i in range(len(BOXES)):
        if data[i][1][0] != 0 and data[i][1][1] != 0 and data[i][1][2] != 0:
            side_positions[i] = [data[i][1][1]*SCALE,
                                 data[i][1][0]*SCALE,
                                 data[i][1][2]*SCALE,
                                 curr_positions[i][3]]

    return side_positions


def decide_position_2(data):
    DELTA = 10
    DELTA_MAX = 25 + DELTA
    DELTA_MIN = 25 - DELTA
    for i in range(len(BOXES)):
        temp = []
        for j in range(len(data[i])):
            if data[i][1][0] != 0 and data[i][1][1] != 0 and data[i][1][2] != 0:
                temp.append([data[i][1][1]*SCALE,
                             data[i][1][0]*SCALE,
                             data[i][1][2]*SCALE,
                             curr_positions[i][3]])
            else:
                temp.append([0, 0, 0])

        is_valid = False
        for j in range(len(temp)):
            if j == 0:
                continue
            sub_x = abs(temp[0][0] - temp[j][0])
            sub_y = abs(temp[0][1] - temp[j][1])
            sub_z = abs(temp[0][2] - temp[j][2])
            if (((sub_x <= DELTA_MAX and sub_x >= DELTA_MIN) or
                 (sub_y <= DELTA_MAX and sub_y >= DELTA_MIN)) and
               sub_z <= DELTA_MAX and sub_z >= DELTA_MIN):
                box_pos = temp[0]
                box_pos[2] -= 25
                is_valid = True
                break

        if is_valid is False:
            for j in range(len(temp)):
                if j == 0:
                    continue
                for m in range(len(temp)):
                    if m <= j:
                        continue

                    sub_x = abs(temp[j][0] - temp[m][0])
                    sub_y = abs(temp[j][1] - temp[m][1])
                    sub_z = abs(temp[j][2] - temp[m][2])
                    if (sub_x <= DELTA_MAX and sub_x >= DELTA_MIN and
                       sub_y <= DELTA_MAX and sub_y >= DELTA_MIN and
                       sub_z <= DELTA):
                        box_pos = []  # TODO
                        is_valid = True
                        break


def decide_position(data, fouth_axis):
    global turn_idx

    for i in range(len(BOXES)):
        if i <= turn_idx or len(data[i]) == 0:
            continue
        temp = []
        for j in range(len(data[i])):
            if data[i][j][0] != 0 and data[i][j][1] != 0 and data[i][j][2] != 0:
                if fouth_axis == -1:
                    fouth_axis = curr_positions[i][3]
                temp.append([data[i][j][1]*SCALE,
                             data[i][j][0]*SCALE,
                             data[i][j][2]*SCALE,
                             fouth_axis])
            else:
                temp.append([])

        # if temp[1] != []:
        #     curr_positions[i] = temp[1]
        #     curr_positions[i][1] += 25
        # else:
        if temp[0] != []:
            curr_positions[i] = temp[0]


def process():
    global curr_positions
    global turn_idx

    test_weight_dir = os.path.join(os.path.dirname(__file__), "..", "training",
                                   WEIGHT_DIR, test_weight_file)
    model = get_model()
    model.load_weights(test_weight_dir)#
    print(model.summary())
    print("")

    models = load_cnn_models()#倒入自动解码模型

    target_state_idx = -1

    message = str(target_state_idx)
    if target_state_idx == -1:
        message = "Init message"
    s.send(message.encode())

    while True:
        print("===========================================================")
        print("")

        is_first_rand = True

        curr_positions = copy.deepcopy(ZERO_POSITIONS)

        if target_state_idx == -1:
            recv_imgs_and_find_keypoints(models)

            data = pickle.loads(s.recv(BUFFER_SIZE), encoding="bytes")
            if data[0] == 0:
                print("Finished")
                break
            # print("Target box positions:", data[1])
            # for i in range(len(BOXES)):
            #     curr_positions[i] = [data[1][i][1]*SCALE,
            #                          data[1][i][0]*SCALE,
            #                          data[1][i][2]*SCALE, 1]
            decide_position(data[1], 1)
            target_state_idx = position_to_state(-1, False, 0)
            target_state_idx = 28

        print("Target state for robot, i.e., last state performed by humnan:",
              target_state_idx)
        message = str(target_state_idx)
        s.send(message.encode())

        draw_curr_state(target_state_idx)

        scenario = one_hot_encode([target_state_idx], N_STATES)
        scenario = scenario.flatten()

        recv_imgs_and_find_keypoints(models)

        data = pickle.loads(s.recv(BUFFER_SIZE), encoding="bytes")
        curr_positions = copy.deepcopy(ZERO_POSITIONS)
        # assign_position(data[1])
        decide_position(data[1], -1)
        print("Initial 3D positions calculated from detected 2D keypoints " +
              "and depth image:")
        print(np.array(curr_positions).astype(int))

        # position_to_image()

        first_state = 0
        curr_state = state_after_moved = 0
        is_first_move = True
        input_states = [0 for _ in range(n_timesteps)]
        output_states = []
        actions = []

        turn_idx = -1
        while True:
            print("===========================================================")
            if is_first_move is False:
                input_states.append(state_after_moved)
                input_states.pop(0)
            target_list = [target_state_idx]
            print("Input states for LSTM: ", input_states, " and ", target_list)

            input_state = np.array(copy.deepcopy(input_states))
            curr_state = input_state[-1]
            input_state = one_hot_encode(input_state, N_STATES)
            input_state = np.array([np.concatenate((input_state[i], scenario))
                                    for i in range(len(input_state))])
            input_state = input_state.reshape(1, n_timesteps,
                                              input_state.shape[1])

            y_predicted = model.predict(input_state, batch_size=1, verbose=0)
            action_predicted = one_hot_decode(y_predicted[0])
            print("Next action predicted by LSTM:",
                  ACTION_IDX_TO_TITLE[action_predicted].upper())
            actions.append(action_predicted)

            if action_predicted == NONE_ACTION_IDX:
                s.send(pickle.dumps([1], protocol=2))
                print("TASK COMPLETED!!!")
                break

            box_idx, next_positions, is_first_rand = action_to_position(
                action_predicted, is_first_move, is_first_rand)
            s.send(pickle.dumps([0,
                                [curr_positions[box_idx][1]/SCALE_F,
                                 curr_positions[box_idx][0]/SCALE_F,
                                 curr_positions[box_idx][2]/SCALE_F],
                                [next_positions[box_idx][1]/SCALE_F,
                                 next_positions[box_idx][0]/SCALE_F,
                                 next_positions[box_idx][2]/SCALE_F]],
                                protocol=2))

            recv_imgs_and_find_keypoints(models)

            data = pickle.loads(s.recv(BUFFER_SIZE), encoding="bytes")
            # assign_position(data[1])
            decide_position(data[1], -1)
            turn_idx += 1
            print("3D positions calculated from detected 2D keypoints " +
                  "and depth image:")
            print(np.array(curr_positions).astype(int))

            state_after_moved = position_to_state(action_predicted,
                                                  is_first_move, first_state)
            curr_state = state_after_moved
            if is_first_move:
                first_state = curr_state
                is_first_move = False

            output_states.append(state_after_moved)

            draw_curr_state(curr_state)

    s.close()


def test_position_to_state():
    position = [[0.440735499064127, 0.0681931372318002, 0.0517574083060026],
                [0.433766648173332, -0.0379426064901053, 0.04711061553098261],
                [0.439887858099407, 0.01673687016591429, 0.05132904804001251]]

    for i in range(len(BOXES)):
        curr_positions[i] = [position[i][1]*SCALE,
                             position[i][0]*SCALE,
                             position[i][2]*SCALE, 1]
    print(curr_positions)
    target_state_idx = position_to_state(-1, False, 0)

    print(target_state_idx)


if __name__ == "__main__":
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((TCP_IP, TCP_PORT))

    turn_idx = -1

    process()
    # test_position_to_state()
