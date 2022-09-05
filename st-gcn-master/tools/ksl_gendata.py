# sys
import os
import sys
import random
import pickle
import json
import argparse
import yaml

# torch
import torch
import numpy as np
from numpy.lib.format import open_memmap
import torch

from collections import Counter

def load_data(label_path, data_num, mode='train'):

    with open(label_path, 'r') as f:
        video_info = json.load(f)
    keys = video_info.keys()

    file_list = []
    label = []
    ground_truth = dict()

    # 불러올 영상 목록과 라벨 저장
    for i, key in enumerate(keys):
        ground_truth[key] = video_info[key]['label_index']

    ground_truth = sorted(ground_truth.items(), key=lambda x: x[1])

    # 라벨 인덱스를 맵핑_________________________________________________________________________
    # mapping = 0
    # for i in range(data_num):
    #     file_list.append(ground_truth[i][0])
    #     label.append(mapping)

    #     if ground_truth[i][0] < ground_truth[i + 1][0]:
    #         mapping += 1
    # ___________________________________________________________________________________________


    for i in range(data_num):
        file_list.append(ground_truth[i][0])
        #label.append(ground_truth[i][1])

    if mode == 'test':
        file_list = [random.choice(file_list) for i in range(int(data_num / 4))]
        #label = [random.choice(label) for i in range(int(data_num / 5))]

    return file_list

#한 영상마다 불림
def get_data(video_json):

    # output shape (C, T, V, M)
    # get data
    with open(video_json, 'r') as f:
        video_info = json.load(f)

    # fill data_numpy
    data_numpy = np.zeros((Channel, Frame, Joint, Person))
    for frame_info in video_info['data']:
        frame_index = frame_info['frame_index'] - 1
        for m, skeleton_info in enumerate(frame_info["skeleton"]):
            if m >= Person:
                break
            pose = skeleton_info['pose']
            score = skeleton_info['score']
            data_numpy[0, frame_index, :, m] = pose[0::2]
            data_numpy[1, frame_index, :, m] = pose[1::2]
            data_numpy[2, frame_index, :, m] = score

    # # centralization
    # data_numpy[0:2] = data_numpy[0:2] - 0.5
    # data_numpy[0][data_numpy[2] == 0] = 0
    # data_numpy[1][data_numpy[2] == 0] = 0

    # get label index
    label = video_info['label_index']

    # # data augmentation
    # if self.random_shift:
    #     data_numpy = tools.random_shift(data_numpy)
    # if self.random_choose:
    #     data_numpy = tools.random_choose(data_numpy, self.window_size)
    # elif self.window_size > 0:
    #     data_numpy = tools.auto_pading(data_numpy, self.window_size)
    # if self.random_move:
    #     data_numpy = tools.random_move(data_numpy)

    return data_numpy, label


def npy_and_pickle(fp, file_list, label_out_path, mode = 'train/'):

    label_list = []
    #npy파일 만들기
    for i, video_name in enumerate(file_list):
        print('processing data:{} ({}/{})'.format(video_name, i + 1, len(file_list)))
        
        video_json = arg.data + '/raw/' + mode + video_name + '.json'
        data_numpy, label = get_data(video_json)
        label_list.append(label)

        # data.npy(test_data.npy) 파일 저장
        if data_numpy is not None:
            fp[i, :, 0:data_numpy.shape[1], :, :] = data_numpy
        else:
            print(i + 1)
            continue
    

    # pkl파일 만들기
    with open(label_out_path, 'wb') as f:
        pickle.dump((file_list, list(label_list)), f)
    
    #class 종류 갯수
    return len(Counter(label_list))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Boolean setting for each pipeline')
    parser.add_argument('--data')
    arg = parser.parse_args()

    Channel = 3
    Frame = 300
    Joint = 18
    Person = 1
    data_num = 6288  # 원하는 데이터 갯수
    # 데이터를 training과 validation으로 분배 - 4:1 비율

    data = arg.data[5:]
    train_label_path = arg.data + '/raw/label_train.json'
    test_label_path = arg.data + '/raw/label_test.json'

    data_out_path = arg.data + '/skeleton/data.npy'
    test_data_out_path = arg.data + '/skeleton/test_data.npy'

    label_out_path = arg.data + '/skeleton/label.pkl'
    test_label_out_path = arg.data + '/skeleton/test_label.pkl'

    # npy for data file initate
    fp_train = open_memmap(
        data_out_path,
        dtype='float32',
        mode='w+',
        shape=(data_num, 3, Frame, 18, Person))

    fp_test = open_memmap(
        test_data_out_path,
        dtype='float32',
        mode='w+',
        shape=(int(data_num / 4), 3, Frame, 18, Person))

    file_list = load_data(train_label_path, data_num)
    test_file_list = load_data(test_label_path, data_num, mode='test')

    print("preprocessing for train set.")
    num_class = npy_and_pickle(fp_train, file_list, label_out_path)

    print("\npreprocessing for test set.")
    npy_and_pickle(fp_test, test_file_list, test_label_out_path, mode='test/')


    with open('config/st_gcn/' + data + '/train.yaml') as y:
        set_class = yaml.load(y, Loader=yaml.FullLoader)
   
    set_class['model_args']['num_class'] = num_class

    with open('config/st_gcn/' + data + '/train.yaml', 'w') as y:
        yaml.dump(set_class, y, default_flow_style=False)

    print("complete preprocessing-2")