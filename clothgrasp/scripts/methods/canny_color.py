#!/usr/env/bin python
import cv2
import numpy as np
from pyquaternion import Quaternion as Quat
from copy import deepcopy

class CannyColorEdgeDetector:
    def __init__(self, crop_dims):
        self.crop_dims = crop_dims

    def predict(self, rgb, lower=150, upper=200):
        # Crop depth image
        row_start, row_end, col_start, col_end, step = self.crop_dims
        rgb_cropped = rgb[row_start:row_end:step, col_start:col_end:step]
        rgb_cropped = cv2.cvtColor(rgb_cropped, cv2.COLOR_RGB2GRAY)

        edged = cv2.Canny(rgb_cropped, lower, upper)

        # Take gradients of depth image
        grad_x = cv2.Scharr(rgb_cropped, cv2.CV_32F, dx=1, dy=0)
        grad_y = cv2.Scharr(rgb_cropped, cv2.CV_32F, dx=0, dy=1)

        return np.stack([edged, grad_x, grad_y], axis=2)