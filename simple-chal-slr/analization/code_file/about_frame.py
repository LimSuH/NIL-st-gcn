import os
import sys
from turtle import color
import numpy as np
from natsort import natsorted
import cv2 
import matplotlib.pyplot as plt

# AUTSL, KETI 두 데이터의 영상 프레임 길이 분포를 plot으로 보여주는 코드

def load_npy(path):
    npy_list = []
    samples = os.listdir(path)

    for i, npy_path in enumerate(samples):
        print('\rnow loading npy files from {}: ({} / {})'.format(path, i + 1, len(samples)), end='')

        data_npy = np.load(os.path.join(path, npy_path))
        npy_list.append(data_npy)
    print(" ")
    return npy_list


def len_frame(npy_list):
    #가져온 npy파일로 프레임 길이를 세기 (frame, keypoit, channel) - .shape[0]
    whole_frame = []

    for i, one_npy in enumerate(npy_list):
        print('\rnow analizing npy files.... ({} / {})'.format(i + 1, len(npy_list)), end='')
        whole_frame.append(one_npy.shape[0])
    print("\n")
    return whole_frame



if __name__ == '__main__':
    # input and output paths
    AUTSL_PATH = '/dataset/AUTSL'
    KETI_PATH = '/dataset/KETI_SignLanguage/Keypoints-MMPOSE'


    # AUTSL_npy = load_npy(AUTSL_PATH + '/train_npy') + load_npy(AUTSL_PATH + '/test_npy') + load_npy(AUTSL_PATH + '/val_npy')
    # print("load success: {} has {} npy files.\n".format(AUTSL_PATH.split('/')[2], len(AUTSL_npy)))

    KETI_npy = load_npy(KETI_PATH)
    print("load success: {} has {} npy files.\n\n".format(KETI_PATH.split('/')[2], len(KETI_npy)))

    # AUTSL_frame = len_frame(AUTSL_npy)
    KETI_frame = len_frame(KETI_npy)

    #show result & draw plot
    print("\n\n\nAnalasis RESULT\n\n[DATASET: KETI]\n AVERAGE: {:.3f}\n VARIACE: {:.3f}\n STANDARD DEVIATION:{:.3f}\n MAX:{:.3f}\n________________________________________________________\n\n".format(
        np.mean(KETI_frame), np.var(KETI_frame), np.std(KETI_frame), np.max(KETI_frame)))

    # print("[DATASET: AUTSL]\n AVERAGE: {:.3f}\n VARIACE: {:.3f}\n STANDARD DEVIATION:{:.3f}\n________________________________________________________\n\n".format(
    #     np.mean(AUTSL_frame), np.var(AUTSL_frame), np.std(AUTSL_frame)))
    

    plt.title('HISTOGRAM')
    #plt.hist(AUTSL_frame, color='blue', alpha=0.4, bins=300, range=[0, np.max(KETI_frame)], label='AUTSL', density=False)
    plt.hist(KETI_frame, color='red', alpha=0.4, bins=300, range=[0, np.max(KETI_frame)], label='KETI', density=False)
    plt.legend()
    plt.savefig('histogram.png')
    plt.show()  



        


