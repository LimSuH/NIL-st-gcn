# Simplified Chal-SLR Repository

A modification and minimum adoption from [the original repository](https://github.com/jackyjsy/CVPR21Chal-SLR)

`/keypoint_estimation`: estimate keypoints from video files to save .npy files
<br>

`/preprocess_label`: collect video file names and matching labels from a dataset directory matching a certain condition

`/SL-GCN`
- `integrate_npys.ipynb`: integrate keypoint .npy files and labels to make train-ready files
- `main.py`: main code to train, finetune, and test models
