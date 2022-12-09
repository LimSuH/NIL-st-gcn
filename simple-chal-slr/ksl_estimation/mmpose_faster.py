import os
import sys
import numpy as np
from natsort import natsorted

import mmcv
from mmpose.apis import (collect_multi_frames, inference_top_down_pose_model,
                         init_pose_model, process_mmdet_results, vis_pose_result)
from mmpose.datasets import DatasetInfo
from mmdet.apis import inference_detector, init_detector

has_mmdet = True

# detector config and checkpoint
det_config = '/home/lab/mmdetection/configs/ssd/ssdlite_mobilenetv2_scratch_600e_coco.py'
det_checkpoint = '/home/lab/.cache/torch/hub/checkpoints/det/ssdlite_mobilenetv2_scratch_600e_coco_20210629_110627-974d9307.pth'
det_checkpoint_url = 'https://download.openmmlab.com/mmdetection/v2.0/ssd/ssdlite_mobilenetv2_scratch_600e_coco/ssdlite_mobilenetv2_scratch_600e_coco_20210629_110627-974d9307.pth'

# wholebody keypoint estimator config and checkpoint
pose_config ='/home/lab/mmpose/configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/hrnet_w48_coco_wholebody_384x288_dark_plus.py'
pose_checkpoint = '/home/lab/.cache/torch/hub/checkpoints/wholebody/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth'
pose_checkpoint_url = 'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_384x288_dark-741844ba_20200812.pth'

# source video path
args_video_path = '../keypoint_estimation/video/KETI_SL_0000002337.avi'

# input and output paths
VIDEO_PATH = '/dataset/KETI_SignLanguage/Video/'
OUTPUT_PATH = '../example/'
try:
    input_path = VIDEO_PATH + sys.argv[1] + '/'
except:
    input_path = VIDEO_PATH

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

# build the pose model from a config file and a checkpoint file
pose_model = init_pose_model(pose_config, pose_checkpoint, device=args_device)

# get dataset info
dataset = pose_model.cfg.data['test']['type']
dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
dataset_info = DatasetInfo(dataset_info)

# get paths
paths = []
names = []
for root, _, fnames in natsorted(os.walk(input_path)):
    for fname in natsorted(fnames):     
        path1 = os.path.join(root, fname) 
        if 'depth' in fname:
            continue
        paths.append(path1)
        names.append(fname)

print(f'Total {len(paths)} files are being pose-estimated...')
paths = paths[:10]

for i, path in enumerate(paths):
    output_npy = OUTPUT_PATH + '{}.npy'.format(names[i])
    # if a file name already exists, skip
    if os.path.exists(output_npy):
        continue
    print(f'({i}/{len(paths)}: {path}')

    # read video
    video = mmcv.VideoReader(path)
    assert video.opened, f'Faild to load video file {path}'

    # loop for keypoint estimation
    keypoint_array = []
    size = (video.width, video.height)
    # print('Running inference...')

    for frame_id, cur_frame in enumerate(mmcv.track_iter_progress(video)):

        mmdet_results = inference_detector(det_model, cur_frame)
        
        person_results = process_mmdet_results(mmdet_results, args_det_cat_id)
        #person_results = [{'bbox': np.array([0, 0, size[0], size[1]])}]
        #print(person_results)

        # test a single image, with a list of bboxes.
        pose_results, returned_outputs = inference_top_down_pose_model(
            pose_model,
            cur_frame,
            person_results,
            bbox_thr=args_bbox_thr,
            format='xyxy',
            dataset=dataset,
            dataset_info=dataset_info,
            return_heatmap=False,
            outputs=None)

        keypoint_array.append(pose_results[0]['keypoints'])

    keypoint_array = np.array(keypoint_array)
    np.save(output_npy, keypoint_array)