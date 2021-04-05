#!/usr/env/bin python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import torch
from unet import unet
from torch.autograd import Variable
import numpy as np
import torchvision.transforms as T
from PIL import Image
import cv2

class ClothEdgeModel:
    def __init__(self, crop_dims, predict_angle, model_path):
        self.predict_angle = predict_angle
        self.model_path = model_path
        self.model_last_updated = os.path.getmtime(self.model_path)
        self.use_gpu = torch.cuda.is_available()
        self.num_gpu = list(range(torch.cuda.device_count()))
        self.crop_dims = crop_dims

        self.transform = T.Compose([T.ToTensor()])

        self.n_class = 3
        # self.n_class = 6
        self.model = unet(n_classes=self.n_class, in_channels=1)
        self.model.load_state_dict(torch.load(self.model_path, map_location='cpu'))
        if self.use_gpu:
            self.model.cuda()

    def update(self):
        """Update the model weights if there is a new version of the weights file"""
        model_updated = os.path.getmtime(self.model_path)
        if model_updated > self.model_last_updated:
            self.model.load_state_dict(torch.load(self.model_path, map_location='cpu'))
            if self.use_gpu:
                self.model.cuda()
            self.model_last_updated = model_updated

    def processHeatMap(self, hm, cmap = plt.get_cmap('jet')):
        resize_transform = T.Compose([T.ToPILImage()])
        hm = torch.Tensor(hm)
        hm = np.uint8(cmap(np.array(hm)) * 255)
        return hm

    def postprocess(self, pred, threshold=100):
        """
        Runs the depth image through the model.
        Returns the dense prediction of corners, outer edges, inner edges, and a three-channel image with all three.
        """
        # pred = np.load('/media/ExtraDrive1/clothfolding/test_data/pred_62_19_01_2020_12:53:16.npy')

        corners = self.processHeatMap(pred[:, :, 0])
        outer_edges = self.processHeatMap(pred[:, :, 1])
        inner_edges = self.processHeatMap(pred[:, :, 2])

        corners = corners[:,:,0]
        corners[corners<threshold] = 0
        corners[corners>=threshold] = 255

        outer_edges = outer_edges[:,:,0]
        outer_edges[outer_edges<threshold] = 0
        outer_edges[outer_edges>=threshold] = 255

        inner_edges = inner_edges[:,:,0]
        inner_edges[inner_edges<threshold] = 0
        inner_edges[inner_edges>=threshold] = 255

        return corners, outer_edges, inner_edges

    def predict(self, image):
        self.model.eval()
        
        row_start, row_end, col_start, col_end, step = self.crop_dims

        image = image[row_start:row_end:step, col_start:col_end:step]
        max_d = np.nanmax(image)
        image[np.isnan(image)] = max_d
        img_depth = Image.fromarray(image, mode='F')
        img_depth = self.transform(img_depth)

        min_I = img_depth.min()
        max_I = img_depth.max()
        img_depth[img_depth<=min_I] = min_I
        img_depth = (img_depth - min_I) / (max_I - min_I)
        
        img_depth = img_depth[np.newaxis, :]

        if self.use_gpu:
            inputs = Variable(img_depth.cuda())
        else:
            inputs = Variable(img_depth)

        outputs = self.model(inputs)
        outputs = torch.sigmoid(outputs)
        output = outputs.data.cpu().numpy()
        N, _, h, w = output.shape
        pred = output.transpose(0, 2, 3, 1)
        pred = pred[0]

        corners, outer_edges, inner_edges = self.postprocess(pred)
        return corners, outer_edges, inner_edges, pred

if __name__ == '__main__':
    im = np.load('/media/ExtraDrive1/clothfolding/data_towel_table/camera_0/depth_467_31_12_2019_12:51:20:230966.npy')
    m = ClothEdgeModel()
    out = m.predict(im)
    # print(out)
