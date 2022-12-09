from cProfile import label
import os
import sys
from turtle import color
import numpy as np
from natsort import natsorted
import cv2 
import matplotlib.pyplot as plt

#AUTSL, KETI x, y 분포 plot로 나타내는 코드

def load_npy(path):
    npy_list = []
    samples = os.listdir(path)
    start_idx = 28565
    samples = samples[start_idx:]
 
    for i, npy_path in enumerate(samples):
        print('\rnow loading npy files from {}: ({} / {})'.format(npy_path, i + start_idx, 33517), end='')
        
        data = np.load(os.path.join(path, npy_path))

        if i == 0 :
            data_npy = np.load(os.path.join(path, npy_path))

            if os.path.isfile('../KETI_merge.npy'):
                data_npy= np.load('/users/suhyeon/GitHub/NIL-st-gcn/simple-chal-slr/KETI_merge.npy')
                
            continue
        
       
        data_npy = np.concatenate((data_npy, data))
        np.save('../KETI_merge.npy', data_npy)

    return data_npy

def separation(data_npy, norm = False, mode = 'KETI'):
    
    x = data_npy[:, :, 0]
    y = data_npy[:, :, 1]

    x = np.reshape(x, (x.shape[0] * x.shape[1], 1))
    y = np.reshape(y, (y.shape[0] * y.shape[1], 1))

    x = np.squeeze(x, axis=1)
    y = np.squeeze(y, axis=1)

    if norm:

        if mode =='AUTSL':
            x = x / 512
            y = y / 512
        else:
            print(x.max(), y.max())
            x = x / 1280
            y = y / 720
            print(x.max(), y.max())

    return x, y
    


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
    KETI_PATH = '/dataset/KETI_SignLanguage/Keypoints-removal'


    # #AUTSL_npy = load_npy(AUTSL_PATH + '/train_npy') + load_npy(AUTSL_PATH + '/test_npy') + load_npy(AUTSL_PATH + '/val_npy')
    #AUTSL_npy = load_npy(AUTSL_PATH + '/val_npy')
    AUTSL_npy = np.load('../save_file/AUTSL_merge.npy')
    print("load success: {} has {} npy files.".format(AUTSL_PATH.split('/')[2], AUTSL_npy.shape))
    AUTSL_x_norm , AUTSL_y_norm = separation(AUTSL_npy, norm=True, mode='AUTSL')
    AUTSL_x , AUTSL_y = separation(AUTSL_npy)

    # KETI_npy = load_npy(KETI_PATH)
    #KETI_npy = load_npy(KETI_PATH + '/val_npy')
    KETI_npy = np.load('../save_file/KETI_merge.npy')
    print("load success: {} has {} npy files.\n".format(KETI_PATH.split('/')[2], KETI_npy.shape))
    KETI_x_norm , KETI_y_norm = separation(KETI_npy, norm=True, mode='KETI')
    KETI_x , KETI_y = separation(KETI_npy)

    print("now draw plots....")

    # plt.subplot(2, 2, 1)
    
    # plt.title('corrdination distribution - normalization')
    # #plt.subplot(1, 2, 1)
    # plt.scatter(AUTSL_x_norm , AUTSL_y_norm, c='blue', label = 'AUTSL', s =1)
    # plt.xlim(0, 1)
    # plt.ylim(0, 1)
    # plt.savefig('../plots/distribution-n-AUTSL.png')
    # plt.clf()

    # plt.title('corrdination distribution - normalization')
    # plt.scatter(KETI_x_norm , KETI_y_norm, c='red', label = 'KETI', s =1)
    # #plt.legend()
    # plt.savefig('../plots/distribution-n-KETI.png')

#____________________________________________________________________________________________
    # plt.subplot(1, 2, 2)
    # #----------------------------------------------------------------
    # plt.title('corrdination distribution')
    # #plt.subplot(1, 2, 1)
    # plt.scatter(AUTSL_x , AUTSL_y, c='blue', label = 'AUTSL',s =1)
    # plt.scatter(KETI_x , KETI_y, c='red', label = 'KETI',s =1)
    # # plt.xlim(0, 1280)
    # # plt.ylim(0, 720)
    # #plt.legend()
    # plt.savefig('../plots/distribution.png')
  

   #____________________________________________________________________________________________________________

    # plt.subplot(2, 2, 1)
    # plt.title('X')
    # plt.hist(AUTSL_x_norm, color='blue', alpha=0.4, bins=300, range=[0, 1], label='AUTSL', density=False)
    
    # plt.subplot(2, 2, 2)
    # plt.hist(KETI_x_norm, color='red', alpha=0.4, bins=300, range=[0, 1], label='KETI', density=False)
    

    # plt.subplot(2, 2, 3)
    # plt.title('Y')
    # plt.hist(AUTSL_y, color='blue', alpha=0.4, bins=300, range=[0, 1], label='AUTSL', density=False)
    
    # plt.subplot(2, 2, 4)
    # plt.hist(KETI_y, color='red', alpha=0.4, bins=300, range=[0, 1], label='KETI', density=False)
    # plt.legend()

    # plt.show()

    # plt.title('scatter - x')
    # plt.scatter()
    # plt.hist(AUTSL_frame, color='blue', alpha=0.4, bins=300, range=[0, np.max(KETI_frame)], label='AUTSL', density=False)
    # plt.hist(KETI_frame, color='red', alpha=0.4, bins=300, range=[0, np.max(KETI_frame)], label='KETI', density=False)
    # plt.legend()
    # plt.savefig('histogram.png')
    # plt.show()  

    # plt.title('scatter - y')
    # plt.hist(AUTSL_frame, color='blue', alpha=0.4, bins=300, range=[0, np.max(KETI_frame)], label='AUTSL', density=False)
    # plt.hist(KETI_frame, color='red', alpha=0.4, bins=300, range=[0, np.max(KETI_frame)], label='KETI', density=False)
    # plt.legend()
    # plt.savefig('histogram.png')
    # plt.show()  



