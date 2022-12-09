import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import os
import argparse
import sys


def calculateAngle(left, center, right, get_degree=False):
    '''
    This function calculates angle between three different landmarks.
    Args:
        landmark1: The first landmark containing the x,y and z coordinates.
        landmark2: The second landmark containing the x,y and z coordinates.
        landmark3: The third landmark containing the x,y and z coordinates.
    Returns:
        angle: The calculated angle between the three landmarks.
    '''

    # Get the required landmarks coordinates.
    x1, y1 = left
    x2, y2 = center
    x3, y3 = right

    # Calculate the angle between the three points
    radian = math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2)

    if get_degree:
        angle = math.degrees(radian)
        # Check if the angle is less than zero.
        if angle < 0:
            # Add 360 to the found angle.
            angle += 360
        return radian, angle
    # Return the calculated angle.
    return radian
    #  dir 전체에 대해서로 코드 변경


def getAngleWithVideo(INPUT_PATH, VIDEO_PATH):
    keypointToAngle = [    
                        (7, 0, 1), (0, 7, 9), (7, 9, 11),
                        (9, 113, 117),
                        (117, 113, 118),(118, 113, 122), (122, 113, 126),(126, 113, 130),
                        (113, 118, 121), (113, 122, 125), (113, 126, 129), (113, 130, 133),

                        (1, 0, 6), (8, 6, 0), (92, 8, 6),
                        (96, 92, 8),
                        (97, 92, 96), (101, 92, 97), (105, 92, 101), (109, 92, 105),
                        (100, 97, 92), (104, 101, 92), (108, 105, 92), (112, 109, 92)
                    ]

    load_path = os.path.join(INPUT_PATH, 'Keypoints-removal')
    video_list = os.listdir(VIDEO_PATH)

    for num, video_file in enumerate(video_list):
        angle_list = []
        npy_file = os.path.join(load_path, video_file) + '.npy'
        save_path = os.path.join(INPUT_PATH, 'Keypoints-angle', video_file) + '.npy'
    
        print('\rnow estimate joint angles with video.... ({} / {})'.format(num + 1, len(video_list)), end='')

        if not os.path.exists(npy_file):# if there is not the keypoint npy, continue
            continue
        elif os.path.exists(save_path):# if there is already angle npy, continue
            continue

        keypoint = np.load(npy_file) # frame x y s
        remov_len = keypoint.shape[0]

        point0 = ((keypoint[:, 5, :] + keypoint[:, 6, :]) / 2).reshape(remov_len, 1, 3) #point0 : a center coordinate of both shoulders
        keypoint = np.concatenate((point0, keypoint), axis=1)

        cap = cv2.VideoCapture(os.path.join(VIDEO_PATH, video_file))
        idx = 0

        total_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        size = (int(width), int(height))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        videoWriter = cv2.VideoWriter(os.path.join('./videos', video_file.split('.')[0]) + '.mp4', fourcc, fps, size)
        #print(keypoint.shape, total_frame)

        # Since the image imports the original image, it is intended to skip the unnecessary frame
        diff = total_frame - remov_len
        
        os.makedirs(os.path.join('./frames', video_file))
        while cap.isOpened():
            ret, image = cap.read()
            angle_one_frame = []

            if not ret:
                #print("error or end of the video.")
                break
            elif idx < diff:
                idx += 1
                continue
            
            i = int(idx - diff)
            for landmarks in keypointToAngle:
                x1, y1 = keypoint[i, landmarks[0], 0], keypoint[i, landmarks[0], 1]  # left
                c1, c2 = keypoint[i, landmarks[1], 0], keypoint[i, landmarks[1], 1]  # center
                x2, y2 = keypoint[i, landmarks[2], 0], keypoint[i, landmarks[2], 1]  # right

                cv2.line(image, (int(c1), int(c2)), (int(x1), int(y1)), (0, 255, 0))
                cv2.line(image, (int(c1), int(c2)), (int(x2), int(y2)), (0, 0, 255))

                rad, angle = calculateAngle((x1, y1), (c1, c2), (x2, y2), get_degree=True)
                _, startAngle = calculateAngle((c1 + 10, c2), (c1, c2), (x1, y1), get_degree=True)
                angle_one_frame.append(rad)

                cv2.ellipse(image, (int(c1), int(c2)), (10, 10), 0, startAngle, startAngle + angle, (0, 255, 255), -1)

            angle_list.append(angle_one_frame)
            #cv2.imshow('result', image)
            videoWriter.write(image)
            cv2.imwrite(os.path.join('./frames', video_file, str(i) + '.jpg'), image)
            idx += 1

            
        cap.release()
        #cv2.destroyAllWindows()

        angle_list = np.array(angle_list)
        print(angle_list.max())
        np.save(save_path, angle_list)

        plt.matshow(angle_list)
        plt.colorbar()

        plt.xlabel('angle index')
        plt.ylabel('frame')

        plt.savefig(os.path.join('./heatmaps', video_file) + '_angle_plot.png')
        plt.close()
        if num == 2:
            break
        
    print(" ")


def getAngleNoVideo(INPUT_PATH):
    angle_list = []
    keypointToAngle = [(7, 0, 1), (9, 7, 0), (115, 9, 7),
                       (117, 113, 9),
                       (118, 113, 117), (122, 113, 118), (126, 113, 122), (130, 113, 126),
                       (121, 118, 113), (125, 122, 113), (129, 126, 113), (133, 130, 113),

                       (1, 0, 6), (0, 6, 8), (6, 8, 92),
                       (8, 92, 96),
                       (96, 92, 97), (97, 92, 101), (101, 92, 105), (105, 92, 109),
                       (92, 97, 100), (92, 101, 104), (92, 105, 108), (92, 109, 112)]

    load_path = os.path.join(INPUT_PATH, 'Keypoints_removal')
    npy_list = os.listdir(load_path)

    for i, npy_file in enumerate(npy_list):
        print('\rnow estimate joint angles.... ({} / {})'.format(i + 1, len(npy_list)), end='')

        keypoint = np.load(os.paht.join(load_path, npy_file))
        frame_len = keypoint.shape[0]
        point0 = ((keypoint[:, 5, :] + keypoint[:, 6, :]) / 2).reshape(frame_len, 1, 3)
        keypoint = np.concatenate((point0, keypoint), axis=1)

        for idx, _ in enumerate(keypoint):
            angle_one_frame = []
            for landmarks in keypointToAngle:
                x1, y1 = keypoint[idx, landmarks[0], 0], keypoint[idx, landmarks[0], 1]  # left
                c1, c2 = keypoint[idx, landmarks[1], 0], keypoint[idx, landmarks[1], 1]  # center
                x2, y2 = keypoint[idx, landmarks[2], 0], keypoint[idx, landmarks[2], 1]  # right

                rad, angle = calculateAngle((x1, y1), (c1, c2), (x2, y2), get_degree=True)
                _, startAngle = calculateAngle((c1 + 10, c2), (c1, c2), (x1, y1), get_degree=True)
                angle_one_frame.append(rad)

            angle_list.append(angle_one_frame)

        angle_list = np.array(angle_list)
        np.save(os.path.join(INPUT_PATH, 'Keypoints_angle', npy_file), angle_list)


def select_option(INPUT_PAHT, make_video = True):
    if make_video:
        getAngleWithVideo(INPUT_PAHT, VIDEO_PATH)
    else:
        getAngleNoVideo(INPUT_PAHT)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Boolean setting for each pipeline')
    parser.add_argument('--inputVideo', type=str, default='/dataset/KETI_SignLanguage/Video/0001~3000')
    args = parser.parse_args()

    INPUT_PATH = '/dataset/KETI_SignLanguage'
    VIDEO_PATH = args.inputVideo

    select_option(INPUT_PATH, VIDEO_PATH)
