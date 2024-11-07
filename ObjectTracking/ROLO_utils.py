# Copyright (c) <2016> <GUANGHAN NING>. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import time
import numpy as np
import math
import cv2


def load_folder(path):
    paths = [os.path.join(path, fn) for fn in next(os.walk(path))[2]]
    paths.sort()
    return paths


def load_dataset_gt(gt_file):
    txtfile = open(gt_file, "r")
    lines = txtfile.read().split('\n')  # '\r\n'
    return lines


def file_to_img(filepath):
    print('Processing ' + filepath)
    img = cv2.imread(filepath)
    return img


def locations_normal(wid, ht, locations):
    #print("location in func: ", locations)
    wid *= 1.0
    ht *= 1.0
    locations[0] *= wid
    locations[1] *= ht
    locations[2] *= wid
    locations[3] *= ht
    return locations


def find_gt_location(lines, id):
    line = lines[id]
    elems = line.split('\t')   # for gt type 2
    if len(elems) < 4:
        elems = line.split(',')  # for gt type 1
    x1 = elems[0]
    y1 = elems[1]
    w = elems[2]
    h = elems[3]
    gt_location = [int(x1), int(y1), int(w), int(h)]
    return gt_location


def find_yolo_location(fold, id):
    paths = [os.path.join(fold, fn) for fn in next(os.walk(fold))[2]]
    paths = sorted(paths)
    path = paths[id-1]
    # print(path)
    yolo_output = np.load(path)
    # print(yolo_output[0][4096:4102])
    yolo_location = yolo_output[0][4097:4101]
    return yolo_location


def find_rolo_location(fold, id):
    filename = str(id) + '.npy'
    path = os.path.join(fold, filename)
    rolo_output = np.load(path)
    return rolo_output


def debug_3_locations(img, gt_location, yolo_location, rolo_location):
    img_cp = img.copy()
    for i in range(3):  # b-g-r channels
        if i == 0:
            location = gt_location
            color = (0, 0, 255)       # red for gt
        elif i == 1:
            location = yolo_location
            color = (255, 0, 0)   # blur for yolo
        elif i == 2:
            location = rolo_location
            color = (0, 255, 0)   # green for rolo
        x = int(location[0])
        y = int(location[1])
        w = int(location[2])
        h = int(location[3])
        if i == 2:
            # if i == 1 or i == 2:
            cv2.rectangle(img_cp, (x-w//2, y-h//2), (x+w//2, y+h//2), color, 2)
            pass
        elif i == 0:
            #cv2.rectangle(img_cp, (x, y), (x+w, y+h), color, 2)
            pass

    return img_cp


def validate_box(box):
    for i in range(len(box)):
        if math.isnan(box[i]):
            box[i] = 0


def iou(box1, box2):
    # Prevent NaN in benchmark results
    validate_box(box1)
    validate_box(box2)

    # change float to int, in order to prevent overflow
    box1 = list(map(int, box1))
    box2 = list(map(int, box2))

    tb = min(box1[0]+0.5*box1[2], box2[0]+0.5*box2[2]) - \
        max(box1[0]-0.5*box1[2], box2[0]-0.5*box2[2])
    lr = min(box1[1]+0.5*box1[3], box2[1]+0.5*box2[3]) - \
        max(box1[1]-0.5*box1[3], box2[1]-0.5*box2[3])
    if tb <= 0 or lr <= 0:
        intersection = 0
        # print "intersection= 0"
    else:
        intersection = tb*lr
    return intersection / (box1[2]*box1[3] + box2[2]*box2[3] - intersection)


def cal_rolo_IOU(location, gt_location):
    location[0] = location[0] - location[2]/2
    location[1] = location[1] - location[3]/2
    loss = iou(location, gt_location)
    return loss


def cal_yolo_IOU(location, gt_location):
    # Translate yolo's box mid-point (x0, y0) to top-left point (x1, y1), in order to compare with gt
    location[0] = location[0] - location[2]/2
    location[1] = location[1] - location[3]/2
    loss = iou(location, gt_location)
    return loss
