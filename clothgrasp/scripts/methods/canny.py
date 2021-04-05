#!/usr/env/bin python
import cv2
import numpy as np
from pyquaternion import Quaternion as Quat
from copy import deepcopy

class CannyEdgeDetector:
    def __init__(self, crop_dims):
        self.crop_dims = crop_dims

    def predict(self, depth, lower=0, upper=25):
        # Crop depth image
        row_start, row_end, col_start, col_end, step = self.crop_dims
        depth_cropped = depth[row_start:row_end:step, col_start:col_end:step]

        # Inpaint holes
        zeros = np.where(depth_cropped == 0)
        mask = np.zeros_like(depth_cropped, np.uint8)
        mask[zeros] = 1
        depth_inpainted = cv2.inpaint(depth_cropped, mask, 3, cv2.INPAINT_NS)

        # Smooth image
        depth_blurred = cv2.GaussianBlur(depth_inpainted, (3, 3), 0)

        blurred_int = (depth_blurred*255).astype(np.uint8)
        edged = cv2.Canny(blurred_int, lower, upper)

        # Take gradients of depth image
        grad_x = cv2.Scharr(depth_blurred, cv2.CV_32F, dx=1, dy=0)
        grad_y = cv2.Scharr(depth_blurred, cv2.CV_32F, dx=0, dy=1)

        return np.stack([edged, grad_x, grad_y], axis=2)