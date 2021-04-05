#!/usr/env/bin python
import cv2
import numpy as np
from pyquaternion import Quaternion as Quat
from copy import deepcopy
import matplotlib.pyplot as plt

class HarrisColorDetector:
    def __init__(self, crop_dims):
        self.crop_dims = crop_dims

    def predict(self, rgb):
        # Crop rgb images
        row_start, row_end, col_start, col_end, step = self.crop_dims
        rgb_cropped = rgb[row_start:row_end:step, col_start:col_end:step]
        gray = cv2.cvtColor(rgb_cropped, cv2.COLOR_RGB2GRAY)

        # Harris detector
        harris = cv2.cornerHarris(gray, 5, 3, 0.04)

        # NMS
        kernel = np.ones((3,3),np.float32)
        harris_dilated = cv2.dilate(harris, kernel)
        max_bin = harris_dilated == harris
        max_bin = harris * (max_bin).astype(np.float)

        # Take gradients of depth image
        grad_x = cv2.Scharr(gray, cv2.CV_32F, dx=1, dy=0)
        grad_y = cv2.Scharr(gray, cv2.CV_32F, dx=0, dy=1)
        grad_x_blur = cv2.GaussianBlur(grad_x, (5, 5), 0)
        grad_y_blur = cv2.GaussianBlur(grad_y, (5, 5), 0)
        _, angles = cv2.cartToPolar(grad_x_blur, grad_y_blur)

        return np.stack([max_bin[:np.newaxis], angles], axis=2)