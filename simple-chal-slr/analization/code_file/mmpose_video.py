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

#mmpose의 좌표값들을 다시 영상으로 나타내는 코드


# input and output paths
INPUT_PATH = '/dataset/KETI_SignLanguage/Video/' + sys.argv[1] + '/'
OUTPUT_PATH = './hrnet_result/'

samples = natsorted(os.listdir(INPUT_PATH))
samples = random.sample(samples, 200)
#samples = random.sample(samples, 10)

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
    INPUT_VIDEO = INPUT_PATH + '{}.avi'.format(path.split('.')[0])
    #INPUT_NPY = '/dataset/KETI_SignLanguage/Keypoints-MMPOSE/' + path + '.npy'
    INPUT_NPY = '/dataset/KETI_SignLanguage/Keypoints/' + path + '.npy'

    # read video
    video = mmcv.VideoReader(INPUT_VIDEO)
    assert video.opened, f'Faild to load video file {INPUT_VIDEO}'
    
    fps = video.fps
    size = (video.width, video.height)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    videoWriter = cv2.VideoWriter(output_name, fourcc, fps, size)

    data_npy = np.load(INPUT_NPY)

    for frame_id, cur_frame in enumerate(mmcv.track_iter_progress(video)):

        pose_results = [{'bbox': np.array([0, 0, size[0], size[1]]), 'keypoints': data_npy[frame_id]}]

        #print('this is npy file shape:', len(pose_results))
        
        #show the results
        vis_frame = vis_pose_result(
            pose_model,
            cur_frame,
            pose_results,
            dataset=dataset,
            dataset_info=dataset_info,
            kpt_score_thr=args_kpt_thr,
            radius=args_radius,
            thickness=args_thickness,
            show=False)

        videoWriter.write(vis_frame)

    videoWriter.release()
