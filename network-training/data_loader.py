# -*- coding: utf-8 -*-

from __future__ import print_function

from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import random
import os
import cv2
import random
from utils import normalize

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import utils
import torchvision.transforms.functional as tf
import torchvision.transforms as T

class TowelDataset(Dataset):

    def __init__(self, root_dir, phase, use_transform=True, datasize=None):
        self.root_dir = root_dir
        self.phase = phase
        self.use_transform = use_transform

        filename = [f for f in os.listdir(self.root_dir) if f.startswith("rgb")]
        self.imgs = filename if datasize is None else filename[0:datasize]
        print(self.imgs)

        if self.phase == 'train':
            self.total_data_num = int(len(self.imgs)/6*4) if len(self.imgs) > 8 else len(self.imgs)
        elif self.phase == 'val':
            self.total_data_num = int(len(self.imgs)/6)
        elif self.phase == 'test':
            self.total_data_num = int(len(self.imgs)/6)

        print("Datapoints: %d" % self.total_data_num)

    def __len__(self):
        return self.total_data_num

    def __getitem__(self, idx):
        if self.phase == 'val':
            idx = idx + self.total_data_num*4
        elif self.phase == 'test':
            idx = idx + self.total_data_num*5

        imidx = self.imgs[idx].split("_")[1].replace(".png", "")
        img_path = os.path.join(self.root_dir, self.imgs[idx])
        depth_path = os.path.join(self.root_dir, imidx+"_depth.npy")

        img_rgb = Image.open(img_path)

        depth_npy = np.load(depth_path)
        depth_npy[np.isnan(depth_npy)] = max_d = np.nanmax(depth_npy)
        img_depth = Image.fromarray(depth_npy, mode='F')

        transform = T.Compose([T.ToTensor()])
        if self.phase == 'test':
            if self.use_transform:
                img_rgb = transform(img_rgb)
                img_depth = transform(img_depth)
                mask = transform(mask)

            img_depth = normalize(img_depth)
            sample = {'rgb': img_rgb, 'X': img_depth}
        else:
            corners_label = Image.open(os.path.join(self.root_dir, imidx+'_labels_red.png'))
            edges_label = Image.open(os.path.join(self.root_dir, imidx+'_labels_yellow.png'))
            inner_edges_label = Image.open(os.path.join(self.root_dir, imidx+'_labels_green.png'))

            if self.use_transform:
                if random.random() > 0.5:
                    img_rgb = tf.hflip(img_rgb)
                    img_depth = tf.hflip(img_depth)
                    corners_label = tf.hflip(corners_label)
                    edges_label = tf.hflip(edges_label)
                    inner_edges_label = tf.hflip(inner_edges_label)
                if random.random() > 0.5:
                    img_rgb = tf.vflip(img_rgb)
                    img_depth = tf.vflip(img_depth)
                    corners_label = tf.vflip(corners_label)
                    edges_label = tf.vflip(edges_label)
                    inner_edges_label = tf.vflip(inner_edges_label)
                if random.random() > 0.9:
                    angle = T.RandomRotation.get_params([-30, 30])
                    img_rgb = tf.rotate(img_rgb, angle, resample=Image.NEAREST)
                    img_depth = tf.rotate(img_depth, angle, resample=Image.NEAREST)
                    corners_label = tf.rotate(corners_label, angle, resample=Image.NEAREST)
                    edges_label = tf.rotate(edges_label, angle, resample=Image.NEAREST)
                    inner_edges_label = tf.rotate(inner_edges_label, angle, resample=Image.NEAREST)
            img_rgb = transform(img_rgb)
            img_depth = transform(img_depth)

            corners_label = transform(corners_label)
            edges_label = transform(edges_label)
            inner_edges_label = transform(inner_edges_label)

            label = torch.cat((corners_label, edges_label, inner_edges_label), 0)
            img_depth = normalize(img_depth)

            sample = {'rgb': img_rgb, 'X': img_depth, 'Y': label}

        return sample

if __name__ == "__main__":
    train_data = TowelDataset(root_dir="/home/jianingq/data_singleim_grasp/camera_3", phase='val', use_transform=True)

    # show a batch
    batch_size = 1
    for i in range(batch_size):
        sample = train_data[i]
        print(i, sample['X'].size())
        print(sample['X'].max(), sample['X'].min(), sample['X'].type())
        print(sample['Y'].max(), sample['Y'].min(), sample['Y'].type())
        print(sample['rgb'].max(), sample['rgb'].min(), sample['rgb'].type())

        a = sample['Y'].numpy()
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                for k in range(a.shape[2]):
                    if a[i,j,k] != 0 and a[i,j,k] != 1:
                        print(a[i,j,k])

    dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=1)

    for i, batch in enumerate(dataloader):
        print(i, batch['X'].size())
    
        # observe 4th batch
        if i == 0:
            plt.figure()
            show_batch(batch)
            plt.axis('off')
            plt.ioff()
            plt.show()
