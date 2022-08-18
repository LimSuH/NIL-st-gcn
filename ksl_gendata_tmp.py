import os
import sys
import pickle
import argparse
import json
import shutil
import time
from types import NoneType

import numpy as np
from numpy.lib.format import open_memmap
import torch

from processor.io import IO
import tools
import tools.utils as utils

import cv2
import pyopenpose as op


# class AIHub_gendata(IO):
# for AIHub subdataset


class KETI_gendata(IO):

    def start(self):
        print("making label.json file...")
        # os.system("python tools/utils/making_label.py")
        num_person_out = 2  # then choose 2 persons with the highest score
        max_frame = 300

        data_path = self.arg.data + '/raw/Video'
        data_out_path = self.arg.data + '/skeleton/data.npy'
        label_path = self.arg.data + '/raw/label.json'
        label_out_path = self.arg.data + '/skeleton/label.pkl'

        # _____________________________________________________________________________________space for checking...____________________________________________________________________
        f = open('video.txt', 'r')
        data = f.read()
        keys = data.split('\n')
        # _____________________________________________________________________________________space for checking...____________________________________________________________________

        with open(label_path, 'r') as f:
            video_info = json.load(f)
        # keys = video_info.keys()
        file_list = []
        label = []
        for key in keys:
            label.append(video_info[key]['label_index'])
            file_list.append(key)

        # initiate
        opWrapper = op.WrapperPython()
        params = dict(model_folder='./models', model_pose='COCO')
        opWrapper.configure(params)
        opWrapper.start()

        for i, video_name in enumerate(file_list):
            input_video = data_path + "/" + video_name + ".avi"
            # input_video = data_path + '/KETI_SL_0000006178.avi'
            fp = open_memmap(
                data_out_path,
                dtype='float32',
                mode='w+',
                shape=(len(file_list), 3, max_frame, 18, num_person_out))

            # pose estimation
            data_numpy = self.pose_estimation(max_frame, input_video, opWrapper)
            print('processing data:{} ({}/{})'.format(video_name, i + 1, len(file_list)))

            if data_numpy is not None:
                fp[i, :, 0:data_numpy.shape[1], :, :] = data_numpy
            else:
                print(i + 1)
                continue

        with open(label_out_path, 'wb') as f:
            pickle.dump((file_list, list(label)), f)

        print("complete preprocessing.")

    def pose_estimation(self, max_frame, input_video, opWrapper):
        self.model.eval()

        video_capture = cv2.VideoCapture(input_video)
        video_length = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        pose_tracker = naive_pose_tracker(data_frame=video_length)

        # pose estimation
        start_time = time.time()
        frame_index = 0
        while True:
            # get image
            ret, orig_image = video_capture.read()
            if orig_image is None:
                break

            source_H, source_W, _ = orig_image.shape
            orig_image = cv2.resize(
                orig_image, (256 * source_W // source_H, 256))
            H, W, _ = orig_image.shape

            # pose estimation
            datum = op.Datum()
            datum.cvInputData = orig_image
            opWrapper.emplaceAndPop(op.VectorDatum([datum]))
            multi_pose = datum.poseKeypoints  # (num_person, num_joint, 3)
            if multi_pose is None:
                continue
            if len(multi_pose.shape) != 3:
                continue

            # normalization
            multi_pose[:, :, 0] = multi_pose[:, :, 0] / W
            multi_pose[:, :, 1] = multi_pose[:, :, 1] / H
            multi_pose[:, :, 0:2] = multi_pose[:, :, 0:2] - 0.5
            multi_pose[:, :, 0][multi_pose[:, :, 2] == 0] = 0
            multi_pose[:, :, 1][multi_pose[:, :, 2] == 0] = 0

            # pose tracking
            pose_tracker.update(multi_pose, frame_index)
            frame_index += 1

            # print('Pose estimation ({}/{}).'.format(frame_index, video_length))

        data_numpy = pose_tracker.get_skeleton_sequence()

        if video_length > max_frame:
            data_numpy = data_numpy[:, -301:-1, :, :]
        return data_numpy

    @staticmethod
    def get_parser(add_help=False):

        # parameter priority: command line > config > default
        parent_parser = IO.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='Demo for Spatial Temporal Graph Convolution Network')

        # region arguments yapf: disable
        parser.add_argument('--data',
                            default='./data/KETI',
                            help='Path to video')
        parser.add_argument('--openpose',
                            default=None,
                            help='Path to openpose')
        parser.add_argument('--model_input_frame',
                            default=128,
                            type=int)
        parser.add_argument('--model_fps',
                            default=30,
                            type=int)
        parser.add_argument('--height',
                            default=1080,
                            type=int,
                            help='height of frame in the output video.')
        parser.set_defaults(
            config='./config/st_gcn/kinetics-skeleton/demo_offline.yaml')
        parser.set_defaults(print_log=False)
        # endregion yapf: enable

        return parser


class naive_pose_tracker():
    """ A simple tracker for recording person poses and generating skeleton sequences.
    For actual occasion, I recommend you to implement a robuster tracker.
    Pull-requests are welcomed.
    """

    def __init__(self, data_frame=128, num_joint=18, max_frame_dis=np.inf):
        self.data_frame = data_frame
        self.num_joint = num_joint
        self.max_frame_dis = max_frame_dis
        self.latest_frame = 0
        self.trace_info = list()

    def update(self, multi_pose, current_frame):
        # multi_pose.shape: (num_person, num_joint, 3)

        if current_frame <= self.latest_frame:
            return

        if len(multi_pose.shape) != 3:
            return

        score_order = (-multi_pose[:, :, 2].sum(axis=1)).argsort(axis=0)
        for p in multi_pose[score_order]:

            # match existing traces
            matching_trace = None
            matching_dis = None
            for trace_index, (trace, latest_frame) in enumerate(self.trace_info):
                # trace.shape: (num_frame, num_joint, 3)
                if current_frame <= latest_frame:
                    continue
                mean_dis, is_close = self.get_dis(trace, p)
                if is_close:
                    if matching_trace is None:
                        matching_trace = trace_index
                        matching_dis = mean_dis
                    elif matching_dis > mean_dis:
                        matching_trace = trace_index
                        matching_dis = mean_dis

            # update trace information
            if matching_trace is not None:
                trace, latest_frame = self.trace_info[matching_trace]

                # padding zero if the trace is fractured
                pad_mode = 'interp' if latest_frame == self.latest_frame else 'zero'
                pad = current_frame - latest_frame - 1
                new_trace = self.cat_pose(trace, p, pad, pad_mode)
                self.trace_info[matching_trace] = (new_trace, current_frame)

            else:
                new_trace = np.array([p])
                self.trace_info.append((new_trace, current_frame))

        self.latest_frame = current_frame

    def get_skeleton_sequence(self):

        # remove old traces
        valid_trace_index = []
        for trace_index, (trace, latest_frame) in enumerate(self.trace_info):
            if self.latest_frame - latest_frame < self.data_frame:
                valid_trace_index.append(trace_index)
        valid_trace_index = valid_trace_index[0:2]
        self.trace_info = [self.trace_info[v] for i, v in enumerate(valid_trace_index)]

        num_trace = len(self.trace_info)
        if num_trace == 0:
            return None

        data = np.zeros((3, self.data_frame, self.num_joint, num_trace))
        for trace_index, (trace, latest_frame) in enumerate(self.trace_info):
            end = self.data_frame - (self.latest_frame - latest_frame)
            d = trace[-end:]
            beg = end - len(d)
            data[:, beg:end, :, trace_index] = d.transpose((2, 0, 1))

        return data

    # concatenate pose to a trace
    def cat_pose(self, trace, pose, pad, pad_mode):
        # trace.shape: (num_frame, num_joint, 3)
        num_joint = pose.shape[0]
        num_channel = pose.shape[1]
        if pad != 0:
            if pad_mode == 'zero':
                trace = np.concatenate(
                    (trace, np.zeros((pad, num_joint, 3))), 0)
            elif pad_mode == 'interp':
                last_pose = trace[-1]
                coeff = [(p + 1) / (pad + 1) for p in range(pad)]
                interp_pose = [(1 - c) * last_pose + c * pose for c in coeff]
                trace = np.concatenate((trace, interp_pose), 0)
        new_trace = np.concatenate((trace, [pose]), 0)
        return new_trace

    # calculate the distance between a existing trace and the input pose

    def get_dis(self, trace, pose):
        last_pose_xy = trace[-1, :, 0:2]
        curr_pose_xy = pose[:, 0:2]

        mean_dis = ((((last_pose_xy - curr_pose_xy) ** 2).sum(1)) ** 0.5).mean()
        wh = last_pose_xy.max(0) - last_pose_xy.min(0)
        scale = (wh[0] * wh[1]) ** 0.5 + 0.0001
        is_close = mean_dis < scale * self.max_frame_dis
        return mean_dis, is_close


if __name__ == "__main__":
    preprocessor = KETI_gendata()
    preprocessor.start()
