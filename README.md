# Simplified Chal-SLR Repository

A modification and minimum adoption from [the original repository](https://github.com/jackyjsy/CVPR21Chal-SLR)

---
## /keypoint_estimation

### 1. Estimate keypoints
Estimate 133 keypoints from raw video files and save to .npy files using HRNet.
- Download the pre-trained whole-body pose model: [Google Drive](https://drive.google.com/file/d/1f_c3uKTDQ4DR3CrwMSI8qdsTKJvKVt7p/view?usp=sharing) and put it in directory `/keypoint_estimation`
- Change VIDEO_PATH and OUTPUT_PATH in `estimate_keypoints.py` to the local video source and the directory to save .npy files, respectively.
- Run `$ python estimate_keypoints.py [sub-directory name]`:
e.g., for KETI data, a sub-directory name could be `0001~3000`, or `28079~30592`
```sh
$ conda activate pytorch
$ python estimate_keypoints.py 0001~3000
```
This command generates .npy files of estimated keypoints in OUTPUT_PATH that correspond to the video files in the sub-directory 0001~3000.

### 2. Generate sample videos
Read random video files from a sub-directory and generated keypoint-overlaid sample videos.
- Change VIDEO_PATH and KEYPOINT_PATH in `generate_sample_videos.py` to the local video source and the directory of estimated keypoint files.
- Run `python generate_sample_videos.py [sub-directory name]` for 3 random video files in sub-directory, or
- Run `python generate_sample_videos.py [sub-directory name] [file_name]` to convert a specific video file in sub-directory.
- The generated video samples are stored in `./video/keypoints`
```sh
$ python generate_sample_videos.py 0001~3000
$ python generate_sample_videos.py 0001~3000 KETI_SL_0000001797.avi
```

### 3. Sample single image keypoint estimation
`estimate_single_image.ipynb` illustrates shapes and dimensions of variables for single image estimation.  
Use this code to learn data stuructues of keypoint estimation codes.