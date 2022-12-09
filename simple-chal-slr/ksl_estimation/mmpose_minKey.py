import os
import sys
import numpy as np
from natsort import natsorted
import cv2 
import random
import mmcv
from mmpose.apis import (collect_multi_frames, inference_top_down_pose_model,
                         init_pose_model, process_mmdet_results, vis_pose_result)
from mmpose.datasets import DatasetInfo
from mmdet.apis import inference_detector, init_detector

has_mmdet = True

# detector config and checkpoint
det_config = '/home/lab/mmpose/demo/mmdetection_cfg/ssdlite_mobilenetv2_scratch_600e_onehand.py'
det_checkpoint = '/home/lab/.cache/torch/hub/checkpoints/det/ssdlite_mobilenetv2_scratch_600e_onehand-4f9f8686_20220523.pth'
det_checkpoint_url = 'https://download.openmmlab.com/mmpose/mmdet_pretrained/ssdlite_mobilenetv2_scratch_600e_onehand-4f9f8686_20220523.pth'

# wholebody keypoint estimator config and checkpoint
pose_config ='/home/lab/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/vipnas_res50_coco_256x192.py'
pose_checkpoint = '/home/lab/.cache/torch/hub/checkpoints/pose/vipnas_res50_coco_256x192-cc43b466_20210624.pth'
pose_checkpoint_url = 'https://download.openmmlab.com/mmpose/top_down/vipnas/vipnas_res50_coco_256x192-cc43b466_20210624.pth'

# wholebody keypoint estimator config and checkpoint
hand_config ='/home/lab/mmpose/configs/hand/2d_kpt_sview_rgb_img/topdown_heatmap/rhd2d/hrnetv2_w18_rhd2d_256x256.py'
hand_checkpoint = '/home/lab/.cache/torch/hub/checkpoints/hand/hrnetv2_w18_rhd2d_256x256-95b20dd8_20210330.pth'
hand_checkpoint_url = 'https://download.openmmlab.com/mmpose/hand/hrnetv2/hrnetv2_w18_rhd2d_256x256-95b20dd8_20210330.pth'

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
hand_model = init_pose_model(hand_config, hand_checkpoint, device=args_device)
pose_model = init_pose_model(pose_config, pose_checkpoint, device=args_device)

# get dataset info
hand_dataset = hand_model.cfg.data['test']['type']
hand_dataset_info = hand_model.cfg.data['test'].get('dataset_info', None)
hand_dataset_info = DatasetInfo(hand_dataset_info)

pose_dataset = pose_model.cfg.data['test']['type']
pose_dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
pose_dataset_info = DatasetInfo(pose_dataset_info)

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
        
paths = random.sample(paths, 200)
print(f'Total {len(paths)} files are being pose-estimated...')

for i, path in enumerate(paths):
    output_npy = OUTPUT_PATH + '{}.npy'.format(names[i])
    # if a file name already exists, skip
    if os.path.exists(output_npy):
        continue
    print(f'({i + 1}/{len(paths)}: {path}')

    # read video
    video = mmcv.VideoReader(path)
    assert video.opened, f'Faild to load video file {path}'

    # loop for keypoint estimation
    keypoint_array = []
    # print('Running inference...')
    fps = video.fps
    size = (video.width, video.height)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_name = OUTPUT_PATH + '{}.mp4'.format(path.split('/')[-1])
    videoWriter = cv2.VideoWriter(output_name, fourcc, fps, size)

    for frame_id, cur_frame in enumerate(mmcv.track_iter_progress(video)):

        mmdet_results = inference_detector(det_model, cur_frame)
        hand_results = process_mmdet_results(mmdet_results, args_det_cat_id)
        person_results = [{'bbox': np.array([0, 0, size[0], size[1]])}]
        #print(person_results)

        # test a single image, with a list of bboxes.
        hand_results, returned_outputs = inference_top_down_pose_model(
            hand_model,
            cur_frame,
            hand_results,
            bbox_thr = args_bbox_thr,
            format='xyxy',
            dataset=hand_dataset,
            dataset_info=hand_dataset_info,
            return_heatmap=False,
            outputs=None)

            #show the results
        # vis_frame = vis_pose_result(
        #     hand_model,
        #     cur_frame,
        #     hand_results,
        #     dataset=hand_dataset,
        #     dataset_info=hand_dataset_info,
        #     kpt_score_thr=args_kpt_thr,
        #     radius=args_radius,
        #     thickness=args_thickness,
        #     show=False)

        pose_results, returned_outputs = inference_top_down_pose_model(
            pose_model,
            cur_frame,
            person_results,
            format='xyxy',
            dataset=pose_dataset,
            dataset_info=pose_dataset_info,
            return_heatmap=False,
            outputs=None)

        # vis_frame = vis_pose_result(
        #     pose_model,
        #     vis_frame,
        #     pose_results,
        #     dataset=pose_dataset,
        #     dataset_info=pose_dataset_info,
        #     kpt_score_thr=args_kpt_thr,
        #     radius=args_radius,
        #     thickness=args_thickness,
        #     show=False)

        # videoWriter.write(vis_frame)

        if len(hand_results) == 2:
            hand = np.concatenate((hand_results[0]['keypoints'], hand_results[1]['keypoints']), axis=0)
            # print(len(pose_results[0]['keypoints']))
        else:
            hand = np.zeros((42,3), dtype=float)
        keypoint_array.append(np.concatenate((pose_results[0]['keypoints'], hand), axis=0))


    #print(len(keypoint_array))
    keypoint_array = np.array(keypoint_array)
    #print(keypoint_array.shape)
    np.save(output_npy, keypoint_array)