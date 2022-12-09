import os
import sys
from turtle import color
import numpy as np
from natsort import natsorted
import cv2 
import matplotlib.pyplot as plt
import pickle as pkl
import mmcv
from mmpose.apis import (collect_multi_frames, inference_top_down_pose_model,
                         init_pose_model, process_mmdet_results, vis_pose_result)
from mmpose.datasets import DatasetInfo
from mmdet.apis import inference_detector, init_detector


# 비디오에서 움직임이 없는 프레임 길이를 plot으로 보여주는 코드

def video_detect(whole_stop, videos, dataset, dataname):

    print(f'Total {len(videos)} files are being pose-estimated...')

    for i, path in enumerate(videos):
    # read video
        video = mmcv.VideoReader(os.path.join(dataset, path))
        assert video.opened, f'Faild to load video file {path}'

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

            if not hand_results:
                whole_stop[frame_id] += 1
                
            np.save(dataname + '_save.npy', whole_stop)
                
            # else:
            #     cv2.imwrite('moving!'+ str(frame_id) + '.png', cur_frame)
        whole_stop[-1] += 1

def check_tendency(datasets, max_frame):

    idx = 0
    whole_stop = np.zeros(max_frame)
    dataname = datasets[0].split('/')[2]

    if os.path.isfile(dataname + '_save.npy'):
        print("save file found. loading data...\n")
        whole_stop = np.load(dataname + '_save.npy')
        idx = int(whole_stop[-1])

    for i, dataset in enumerate(datasets):
        videos = natsorted(os.listdir(dataset))

        if dataname == 'AUTSL':
            videos= videos[::2]

        if idx > len(videos):
            idx = idx - len(videos)
            continue

        idx = 0

        print('\nnow check from {}'.format(dataset))
        video_detect(whole_stop, videos, dataset, dataname)

    return whole_stop

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


    # build the detection model from a config file and a checkpoint file
    det_model = init_detector(det_config, det_checkpoint, device=args_device)
    hand_model = init_pose_model(hand_config, hand_checkpoint, device=args_device)
    
    # get dataset info
    hand_dataset = hand_model.cfg.data['test']['type']
    hand_dataset_info = hand_model.cfg.data['test'].get('dataset_info', None)
    hand_dataset_info = DatasetInfo(hand_dataset_info)

    # input and output paths
    AUTSL_PATH = '/dataset/AUTSL'
    KETI_PATH = '/dataset/KETI_SignLanguage/Video'
    
    AUTSL_DIR = ['/dataset/AUTSL/train', '/dataset/AUTSL/test', '/dataset/AUTSL/val']
    KETI_DIR = [os.path.join(KETI_PATH, dir) for dir in natsorted(os.listdir(KETI_PATH))]

    AUTSL_max = 600
    KETI_max = 600

    #비디오를 갖고와서 하나씩 핸드 디텍션을 하고 손 감지가 안되는 프레임의 정보를 추출
    KETI_trend = check_tendency(KETI_DIR, KETI_max)
    AUTSL_trend = check_tendency(AUTSL_DIR, AUTSL_max)



    plt.title('NO MOVE FRAME')
    plt.hist(AUTSL_trend[:-1], color='blue', alpha=0.4, bins=100, range=[0, 100], label='AUTSL', density=False)
    plt.hist(KETI_trend[:-1], color='red', alpha=0.4, bins=100, range=[0, 100], label='KETI', density=False)
    plt.legend()
    plt.savefig('none_move_frame.png')
    plt.show()  