from __future__ import print_function

import os
import os.path
from random import random

import h5py
import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader

from point_util import rotate_pts, jitter_pts, translate_pts

DATA_DIR = './shapenet16/'
MODEL_DIR = './modelnet40/'

def load_h5(h5_filename):
    f = h5py.File(h5_filename, 'r')
    data = f['data'][:]
    label = f['label'][:]

    return data, label

def read_h5_dataset(name, num):
    pts_arr, label_arr = [], []

    for i in range(num):
        pts, label = load_h5(os.path.join(DATA_DIR, f'ply_data_{name}{i}.h5'))
        pts_arr.append(pts)
        label_arr.append(label)

    pts = np.vstack(pts_arr)
    label = np.vstack(label_arr)

    return pts, label

def read_model_h5_dataset(name, num):
    pts_arr, label_arr = [], []

    for i in range(num):
        pts, label = load_h5(os.path.join(MODEL_DIR, f'ply_data_{name}{i}.h5'))
        pts_arr.append(pts)
        label_arr.append(label)

    pts = np.vstack(pts_arr)
    label = np.vstack(label_arr)

    return pts, label

class Shape16Dataset(data.Dataset):
    def __init__(self, mode='train', augment_flag=True):
        self.mode = mode
        self.augment_flag = augment_flag

        if mode == "train":
            pts, seg = read_h5_dataset('train', 6)
            pts_cd_dis = np.load('./data/distance/train_mean_distance_cd.npy')
            pts_emd_dis = np.load('./data/distance/train_mean_distance_emd.npy')
            s_index = np.load('./data/S_index/trainset-object_density.npy')
        else:
            pts, seg = read_h5_dataset('test', 2)
            pts_cd_dis = np.load('./data/distance/test_mean_distance_cd.npy')
            pts_emd_dis = np.load('./data/distance/test_mean_distance_emd.npy')
            s_index = np.load('./data/S_index/testset-object_density.npy')

        self.pts, self.seg = pts, seg
        self.pts_cd_dis, self.pts_emd_dis = pts_cd_dis, pts_emd_dis
        self.s_index = s_index

        print('Dataset size', len(self.pts), len(pts_cd_dis), len(pts_emd_dis))

    def __getitem__(self, index):
        pts = self.pts[index]
        seg = self.seg[index]
        cd_dis = self.pts_cd_dis[index]
        emd_dis = self.pts_emd_dis[index]
        s_index = self.s_index[index]
        
        # data augmentation
        if self.augment_flag and random() > 0.5 and self.mode == "train":
            pts = jitter_pts(pts)
        if self.augment_flag and random() > 0.5 and self.mode == "train":
            pts = rotate_pts(pts)
        if self.augment_flag and random() > 0.5 and self.mode == "train":
            pts = translate_pts(pts)

        pts = torch.from_numpy(pts)
        pts = pts.transpose(1, 0)

        seg = torch.from_numpy(seg)

        return pts, seg, cd_dis, emd_dis, s_index

    def __len__(self):
        return len(self.pts)

class ModelNet40Dataset(data.Dataset):
    def __init__(self, mode='train', augment_flag=True):
        self.mode = mode
        self.augment_flag = augment_flag

        if mode == "train":
            pts, seg = read_model_h5_dataset('train', 5)
        else:
            pts, seg = read_model_h5_dataset('test', 2)

        self.pts, self.seg = pts, seg

        print('Dataset size', len(self.pts))

    def __getitem__(self, index):
        pts = self.pts[index]
        seg = self.seg[index]

        # data augmentation
        if self.augment_flag and random() > 0.5 and self.mode == "train":
            pts = jitter_pts(pts)
        if self.augment_flag and random() > 0.5 and self.mode == "train":
            pts = rotate_pts(pts)
        if self.augment_flag and random() > 0.5 and self.mode == "train":
            pts = translate_pts(pts)

        pts = torch.from_numpy(pts)
        pts = pts.transpose(1, 0)

        seg = torch.from_numpy(seg)

        return pts, seg

    def __len__(self):
        return len(self.pts)

if __name__ == '__main__':
    test_dataset = Shape16Dataset(mode='test', augment_flag=True)
    testdataloader = DataLoader(test_dataset, batch_size=1,
                                shuffle=False, num_workers=4)

    from tqdm import tqdm

    while True:
        for full, part in tqdm(test_dataset):
            continue
        break
