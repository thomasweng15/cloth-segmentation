#!/usr/env/bin python
# import matplotlib.pyplot as plt
import cv2
import numpy as np
from pyquaternion import Quaternion as Quat
from copy import deepcopy

class DepthGradDetector:
    def predict(self, depth):
        # Crop depth image
        row_start = 250
        row_end = 650
        col_start = 500
        col_end = 900
        depth_cropped = depth[row_start:row_end, col_start:col_end]

        # Inpaint holes
        zeros = np.where(depth_cropped == 0)
        mask = np.zeros_like(depth_cropped, np.uint8)
        mask[zeros] = 1
        depth_inpainted = cv2.inpaint(depth_cropped, mask, 3, cv2.INPAINT_NS)

        # Smooth image
        depth_blurred = cv2.GaussianBlur(depth_inpainted, (3, 3), 0)

        # Take gradients of depth image
        grad_x = cv2.Scharr(depth_blurred, cv2.CV_32F, dx=1, dy=0)
        # grad_x = cv2.Sobel(depth_blurred, cv2.CV_32F, dx=1, dy=0, ksize=7)
        grad_y = cv2.Scharr(depth_blurred, cv2.CV_32F, dx=0, dy=1)
        # grad_y = cv2.Sobel(depth_blurred, cv2.CV_32F, dx=0, dy=1, ksize=7)

        abs_grad_x = cv2.convertScaleAbs(grad_x, alpha=255/grad_x.max())
        abs_grad_y = cv2.convertScaleAbs(grad_y, alpha=255/grad_y.max())
        grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

        return np.stack([grad, grad_x, grad_y], axis=2)