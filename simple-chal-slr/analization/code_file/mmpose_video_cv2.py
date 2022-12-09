import os
import sys
import numpy as np
from natsort import natsorted
import random
import cv2 
import mmcv
from mmpose.apis import (collect_multi_frames, inference_top_down_pose_model,
                         init_pose_model, process_mmdet_results, vis_pose_result)
from mmpose.datasets import DatasetInfo
from mmdet.apis import inference_detector, init_detector



# mmpose의 결과물을 다시 영상에 나타내는코드(cv2사용)
# input and output paths
INPUT_PATH = '/dataset/AUTSL/train/'
OUTPUT_PATH = './ksl_estimation'

samples = os.listdir(INPUT_PATH)
depth = True

while depth:
    samples = random.sample(samples, 1)
    print("\r{}".format(samples), end='')
    if "depth" not in samples:
        depth = False

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

## wholebody keypoint estimator config and checkpoint
pose_config ='/home/lab/mmpose/configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/hrnet_w48_coco_wholebody_384x288_dark_plus.py'
pose_checkpoint = '/home/lab/.cache/torch/hub/checkpoints/wholebody/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth'
pose_checkpoint_url = 'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_384x288_dark-741844ba_20200812.pth'

pose_model = init_pose_model(pose_config, pose_checkpoint, device=args_device)

dataset = pose_model.cfg.data['test']['type']
dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
dataset_info = DatasetInfo(dataset_info)


print(f'Total {len(samples)} files are being turned into video...')
#print(samples)
for i, path in enumerate(samples):
    output_name = OUTPUT_PATH + '{}.mp4'.format(path.split('.')[0])
    INPUT_VIDEO = INPUT_PATH + '{}.mp4'.format(path.split('.')[0])
    INPUT_NPY = '/dataset/AUTSL/train_npy/' + path + '.npy'

    # # read video
    # video = mmcv.VideoReader(INPUT_VIDEO)
    # assert video.opened, f'Faild to load video file {INPUT_VIDEO}'

    # make video by opencv
    cap = cv2.VideoCapture(INPUT_VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    size = (int(width), int(height))
    
    # fps = video.fps
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    videoWriter = cv2.VideoWriter(output_name, fourcc, fps, size)

    data_npy = np.load(INPUT_NPY)
    frame_id = 0
    while cap.isOpened():
        success, image = cap.read()
        image = cv2.flip(image, 1)
        if not success:
            break

        for keypoint in data_npy[frame_id]:
            cv2.circle(image, (int(keypoint[0]), int(keypoint[1])), 3, (0, 0, 255), -1)
        
        videoWriter.write(image)
        frame_id +=1

    videoWriter.release()
