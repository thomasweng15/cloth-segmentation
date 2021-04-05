#!/usr/env/bin python
import cv2
import numpy as np
from pyquaternion import Quaternion as Quat
from copy import deepcopy

class HarrisDetector:
    def __init__(self, crop_dims):
        self.crop_dims = crop_dims

    def predict(self, depth, thresh=0.01):
        # Crop depth image
        row_start, row_end, col_start, col_end, step = self.crop_dims
        depth_cropped = depth[row_start:row_end:step, col_start:col_end:step]

        # Inpaint holes
        zeros = np.where(depth_cropped == 0)
        mask = np.zeros_like(depth_cropped, np.uint8)
        mask[zeros] = 1
        depth_inpainted = cv2.inpaint(depth_cropped, mask, 3, cv2.INPAINT_NS)

        # Harris detector
        depth_harris = cv2.cornerHarris(depth_inpainted, 5, 3, 0.04)

        # NMS
        kernel = np.ones((3,3),np.float32)
        depth_harris_dilated = cv2.dilate(depth_harris, kernel)
        max_bin = depth_harris_dilated == depth_harris
        max_bin = depth_harris * (max_bin).astype(np.float) 

        # Take gradients of depth image
        grad_x = cv2.Scharr(depth_inpainted, cv2.CV_32F, dx=1, dy=0)
        grad_y = cv2.Scharr(depth_inpainted, cv2.CV_32F, dx=0, dy=1)
        grad_x_blur = cv2.GaussianBlur(grad_x, (5, 5), 0)
        grad_y_blur = cv2.GaussianBlur(grad_y, (5, 5), 0)
        _, angles = cv2.cartToPolar(grad_x_blur, grad_y_blur)

        return np.stack([max_bin, angles], axis=2)