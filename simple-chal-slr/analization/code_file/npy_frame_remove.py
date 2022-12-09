import os
import sys
from turtle import color
import numpy as np
from natsort import natsorted
import cv2 
import json
import matplotlib.pyplot as plt
import pickle as pkl
import mmcv
from mmpose.apis import (collect_multi_frames, inference_top_down_pose_model,
                         init_pose_model, process_mmdet_results, vis_pose_result)
from mmpose.datasets import DatasetInfo
from mmdet.apis import inference_detector, init_detector

# npy파일에서 움직임이 없는 프레임을 잘라내는 코드

#이미 프레임이 잘린 비디오가 있어 그 프레임 만큼 npy파일을 자름
def video_remove(videos, dataset):
    print(f'Total {len(videos)} files are being pose-estimated...')
    global skip_num
    for i, path in enumerate(videos):

        print('\rnow removing frames.... ({} / {})'.format(i + 1, len(videos)), end='')
        
        npy_path = '/dataset/KETI_SignLanguage/Keypoints-MMPOSE/' + path + '.npy'
        output_dir = '/dataset/KETI_SignLanguage/Keypoints-removal/' + path + '.npy'

        if os.path.exists(output_dir):
            skip_num +=1
            continue
        elif not os.path.exists(npy_path):
            skip_num +=1
            continue
        
        sFiles.append(npy_path)
        exist_npy = np.load(npy_path)
        video_path = os.path.join(dataset, path)
        video = mmcv.VideoReader(video_path)
        assert video.opened, f'Faild to load video file {path}'

        start_frame = len(video)
        exist_npy = exist_npy[-start_frame : ]
        np.save(output_dir, exist_npy)

# 프레임을 자른 비디오 파일이 없어 새로 디텍션 후 프레임 자르기 필요
def detect_remove(videos, dataset):
    print(f'Total {len(videos)} files are being pose-estimated...')
    global skip_num
    for i, path in enumerate(videos):

        npy_path = '/dataset/KETI_SignLanguage/Keypoints-MMPOSE/' + path + '.npy'
        output_dir = '/dataset/KETI_SignLanguage/Keypoints-removal/' + path + '.npy'
    
        if os.path.exists(output_dir):
            skip_num +=1
            print("\rThe Video has been already processed. skipped...", end ='')
            continue
        elif not os.path.exists(npy_path):
            skip_num +=1
            print("\rThe Video does not have npy file. skipped...", end ='')
            continue
        
        exist_npy = np.load(npy_path)
        video_path = os.path.join(dataset, path)
        video = mmcv.VideoReader(video_path)
        assert video.opened, f'Faild to load video file {path}'

        #print('\rnow removing frames.... ({} / {})'.format(i + 1, len(videos)), end='')

        #새로 디텍션이 필요
        sFiles.append(npy_path)
        start_frame = 0
        for frame_id, cur_frame in enumerate(mmcv.track_iter_progress(video)):
            print(' ({} / {})'.format(i + 1, len(videos)), end=' ')
            mmdet_results = inference_detector(det_model, cur_frame)
            detect_results = process_mmdet_results(mmdet_results, args_det_cat_id)

            hand_results, returned_outputs = inference_top_down_pose_model(
            hand_model,
            cur_frame,
            detect_results,
            bbox_thr = args_bbox_thr,
            format='xyxy',
            dataset=hand_dataset,
            dataset_info=hand_dataset_info,
            return_heatmap=False,
            outputs=None)

            if hand_results:
                start_frame = frame_id
                break

        exist_npy = exist_npy[start_frame : ]
        np.save(output_dir, exist_npy)
        print(" ")

def mul_dir(removalPath, non_removalPath):

    videos = natsorted(os.listdir(removalPath))

    print('\nnow check from {}'.format(removalPath))
    video_remove(videos, removalPath)

    for i, dataset in enumerate(non_removalPath):
        videos = natsorted(os.listdir(dataset))

        print('\nnow check from {}'.format(dataset))
        detect_remove(videos, dataset)


if __name__ == '__main__':

    # detector config and checkpoint
    det_config = '/home/lab/mmpose/demo/mmdetection_cfg/ssdlite_mobilenetv2_scratch_600e_onehand.py'
    det_checkpoint = '/home/lab/.cache/torch/hub/checkpoints/det/ssdlite_mobilenetv2_scratch_600e_onehand-4f9f8686_20220523.pth'
    det_checkpoint_url = 'https://download.openmmlab.com/mmpose/mmdet_pretrained/ssdlite_mobilenetv2_scratch_600e_onehand-4f9f8686_20220523.pth'

    # hand keypoint estimator config and checkpoint
    hand_config ='/home/lab/mmpose/configs/hand/2d_kpt_sview_rgb_img/topdown_heatmap/rhd2d/hrnetv2_w18_rhd2d_256x256.py'
    hand_checkpoint = '/home/lab/.cache/torch/hub/checkpoints/hand/hrnetv2_w18_rhd2d_256x256-95b20dd8_20210330.pth'
    hand_checkpoint_url = 'https://download.openmmlab.com/mmpose/hand/hrnetv2/hrnetv2_w18_rhd2d_256x256-95b20dd8_20210330.pth'

    # other arguments
    args_show = False
    args_out_video_root = ''
    args_device = 'cuda:0'
    args_det_cat_id = 1
    args_bbox_thr = 0.3
    args_kpt_thr = 0.3
    args_radius = 3
    args_thickness = 1
    args_use_multi_frames = False
    args_online = False
    global skip_num
    skip_num = 0
    global sFiles
    sFiles = []

    # build the detection model from a config file and a checkpoint file
    det_model = init_detector(det_config, det_checkpoint, device=args_device)
    hand_model = init_pose_model(hand_config, hand_checkpoint, device=args_device)
    
    # get dataset info
    hand_dataset = hand_model.cfg.data['test']['type']
    hand_dataset_info = hand_model.cfg.data['test'].get('dataset_info', None)
    hand_dataset_info = DatasetInfo(hand_dataset_info)

    # input and output paths
    REMOVAL = '/dataset/KETI_SignLanguage/removal'
    NONE_REMOVAL = [os.path.join('/dataset/KETI_SignLanguage/Video', dir) for dir in natsorted(os.listdir('/dataset/KETI_SignLanguage/Video'))]
    #KETI_DIR = [os.path.join(KETI_PATH, dir) for dir in natsorted(os.listdir(KETI_PATH))]

    #비디오를 갖고와서 하나씩 핸드 디텍션을 하고 손 감지가 안되는 프레임의 정보를 추출
    KETI_trend = mul_dir(REMOVAL, NONE_REMOVAL)

    with open('skipped.json', 'w') as outfile:
        json.dump(sFiles, outfile)
    