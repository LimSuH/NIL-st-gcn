import os
import sys
import random

import cv2
import numpy as np
from utils import plot_pose

VIDEO_PATH = '/hdd1/dataset/KETI_SignLanguage/Video/'
KEYPOINT_PATH = '/hdd1/dataset/KETI_SignLanguage/Keypoints/'
OUTPUT_PATH = './video/keypoints/'

sub_path = sys.argv[1]
if len(sys.argv) == 3:
    list_files = [sys.argv[2]]
elif len(sys.argv) == 2:
    list_files = random.choices(os.listdir(VIDEO_PATH + sub_path), k=3)

size = (512, 512)
for file_name in list_files:
    print(file_name)
    out = cv2.VideoWriter(OUTPUT_PATH + 'keypoints_' + file_name + '.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, size)

    keypoint_file = os.path.join(KEYPOINT_PATH, file_name + '.npy')
    keypoints = np.load(keypoint_file)
    
    video_file = os.path.join(VIDEO_PATH + sub_path, file_name)
    cap = cv2.VideoCapture(video_file)

    t = 0
    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break
        
        # cropping to h x h
        frame_height, frame_width = img.shape[:2]
        margin = int((frame_width - frame_height) / 2)
        img = img[:, margin : margin + frame_height]

        # resize to 512 x 512
        img = cv2.resize(img, size)
        frame_height, frame_width = img.shape[:2]

        # flip
        img = cv2.flip(img, flipCode=1)

        # create pose-estimated image
        pred = keypoints[t]
        img = plot_pose(img, pred)

        # write to video file
        out.write(img)
        t += 1

    out.release()