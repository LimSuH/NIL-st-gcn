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


# 프레임을 자른 비디오 파일이 없어 새로 디텍션 후 프레임 자르기 필요
def detect_remove(videos, dataset):
    print(f'Total {len(videos)} files are being pose-estimated...')
    global skip_num
    for i, path in enumerate(videos):

        npy_path = '/dataset/KETI_SignLanguage/Keypoints-MMPOSE/' + path + '.npy'
        output_dir = '/dataset/KETI_SignLanguage/Keypoints-removal/' + path + '.npy'

        # if os.path.exists(output_dir):
        #     skip_num += 1
        #     print("\rThe Video has been already processed. skipped...", end='')
        #     continue
        # elif not os.path.exists(npy_path):
        #     skip_num += 1
        #     print("\rThe Video does not have npy file. skipped...", end='')
        #     continue
        
        # for test
        npy_path = '/dataset/KETI_SignLanguage/Keypoints-removal/KETI_SL_0000000396.avi.npy'
        output_dir = './test'


        exist_npy = np.load(npy_path)
        video_path = os.path.join(dataset, path)

        #for test
        video_path = '/dataset/KETI_SignLanguage/Video/0001~3000/KETI_SL_0000000396.avi'
        video = mmcv.VideoReader(video_path)

        # 비디오와 npy 뒤집어 뒤쪽부터 삭제
        exist_npy = exist_npy[::-1]
        video = video[::-1]

        #assert video.opened, f'Faild to load video file {path}'

        # print('\rnow removing frames.... ({} / {})'.format(i + 1, len(videos)), end='')

        # 새로 디텍션이 필요
        sFiles.append(npy_path)
        start_frame = 0
        for frame_id, cur_frame in enumerate(mmcv.track_iter_progress(video)):
            print(' ({} / {})'.format(i + 1, len(videos)), end=' ')
            mmdet_results = inference_detector(det_model, cur_frame)
            detect_results = process_mmdet_results(mmdet_results, args_det_cat_id)

            #mmpose hand estimation을 이용한 손이 보이는지 여부 판정
            #inference_top_down_pose_model은 hand detect 영역(detect_results)안에서 hand keypoint를 반환함
            #만약 손이 검출되지 않아 제대로 된 hand keypoint를 얻지 못했을 경우, 빈 리스트를 반환
            hand_results, returned_outputs = inference_top_down_pose_model(
                hand_model,
                cur_frame,
                detect_results,
                bbox_thr=args_bbox_thr,
                format='xyxy',
                dataset=hand_dataset,
                dataset_info=hand_dataset_info,
                return_heatmap=False,
                outputs=None)

            #hand_results가 존재한다는 것은 유효한 hand keypoint를 반환 받았음을 의미 = 손이 보인다고 판정
            if hand_results:
                start_frame = frame_id
                break

        exist_npy = exist_npy[start_frame:]

        #뒤집힌 영상정보 원상복귀
        exist_npy = exist_npy[::-1]
        video = video[::-1]

        for i, frame in enumerate(mmcv.track_iter_progress(video)):
            cv2.imwrite('./test' + str(i) + '.png', frame)
        np.save(output_dir, exist_npy)
        print(" ")

        #for test
        break

def mul_dir(removalPath):
    videos = natsorted(os.listdir(removalPath))

    print('\nnow check from {}'.format(removalPath))
    detect_remove(videos, removalPath)


if __name__ == '__main__':
    # detector config and checkpoint
    det_config = '/home/lab/mmpose/demo/mmdetection_cfg/ssdlite_mobilenetv2_scratch_600e_onehand.py'
    det_checkpoint = '/home/lab/.cache/torch/hub/checkpoints/det/ssdlite_mobilenetv2_scratch_600e_onehand-4f9f8686_20220523.pth'
    det_checkpoint_url = 'https://download.openmmlab.com/mmpose/mmdet_pretrained/ssdlite_mobilenetv2_scratch_600e_onehand-4f9f8686_20220523.pth'

    # hand keypoint estimator config and checkpoint
    hand_config = '/home/lab/mmpose/configs/hand/2d_kpt_sview_rgb_img/topdown_heatmap/rhd2d/hrnetv2_w18_rhd2d_256x256.py'
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
    NONE_REMOVAL = [os.path.join('/dataset/KETI_SignLanguage/Video', dir) for dir in
                    natsorted(os.listdir('/dataset/KETI_SignLanguage/Video'))]
    # KETI_DIR = [os.path.join(KETI_PATH, dir) for dir in natsorted(os.listdir(KETI_PATH))]

    # 비디오를 갖고와서 하나씩 핸드 디텍션을 하고 손 감지가 안되는 프레임의 정보를 추출
    KETI_trend = mul_dir(REMOVAL)

    with open('skipped.json', 'w') as outfile:
        json.dump(sFiles, outfile)
