#!/usr/env/bin python
import cv2
import numpy as np
from copy import deepcopy

class GroundTruth:
    def __init__(self, crop_dims):
        self.crop_dims = crop_dims

    def predict(self, rgb):
        row_start, row_end, col_start, col_end, step = self.crop_dims
        rgb = rgb[row_start:row_end:step, col_start:col_end:step, :]
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        img_hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
        
        # Range for lower red
        lower_red = np.array([0,75,70])
        upper_red = np.array([5,255,255])
        mask1 = cv2.inRange(img_hsv, lower_red, upper_red)
            
        # Range for red color
        lower_red = np.array([100,75,100])
        upper_red = np.array([180,255,255])
        mask2 = cv2.inRange(img_hsv,lower_red,upper_red)
            
        # Generating the final mask to detect red color
        labels_red = mask1 + mask2

         # -------------------------------------------

        #defining the range of Yellow color
        yellow_lower = np.array([20,75,150],np.uint8)
        yellow_upper = np.array([32,255,255],np.uint8)
        labels_yellow = cv2.inRange(img_hsv, yellow_lower, yellow_upper)

        # --------------------------------------------

        #defining the range of green color
        green_lower = np.array([32,50,70],np.uint8)
        green_upper = np.array([70,255,255],np.uint8)
        labels_green = cv2.inRange(img_hsv, green_lower, green_upper)

        # --------------------------------------------	

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        labels_red = cv2.morphologyEx(labels_red, cv2.MORPH_OPEN, kernel)
        labels_red = cv2.morphologyEx(labels_red, cv2.MORPH_CLOSE, kernel)

        labels_yellow = cv2.morphologyEx(labels_yellow, cv2.MORPH_OPEN, kernel)
        labels_yellow = cv2.morphologyEx(labels_yellow, cv2.MORPH_CLOSE, kernel)

        labels_green = cv2.morphologyEx(labels_green, cv2.MORPH_OPEN, kernel)
        labels_green = cv2.morphologyEx(labels_green, cv2.MORPH_CLOSE, kernel)

        pred = np.stack([labels_red, labels_yellow, labels_green], axis=-1).astype(np.uint8)
        return pred


