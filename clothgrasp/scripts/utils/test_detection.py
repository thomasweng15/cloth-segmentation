#!/usr/bin/env python
import argparse
import rospy
import os
import cv2
import actionlib
import numpy as np
from datetime import datetime
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from detection_visualizer import DetectionVisualizer
from clothgrasp.msg import ExecuteGraspAction, ExecuteGraspGoal, MoveHomeAction, MoveHomeGoal
from clothgrasp.srv import DetectEdge, SelectGrasp, PlanGrasp

class ClothGrasper:
    def __init__(self):
        rospy.init_node('clothgrasper')

        rospy.set_param('break_on_error', False)
        self.base_path = rospy.get_param('base_save_path')
        self.detection_method = rospy.get_param('detection_method')
        self._init_vis()

        self.bridge = CvBridge()

    def _init_vis(self):
        self.crop_dims = rospy.get_param('crop_dims') if self.detection_method == 'network' or self.detection_method == 'groundtruth' else rospy.get_param('crop_dims_baselines')
        self.visualizer = DetectionVisualizer(self.detection_method, self.crop_dims)

    def _call_detectedge_service(self):
        rospy.wait_for_service('detect_edges')
        detect_edge = rospy.ServiceProxy('detect_edges', DetectEdge)
        return detect_edge()

    def _aggregate_data(self, detectedge_response):
        corners = None
        outer_edges = None
        inner_edges = None

        detection_method = rospy.get_param('detection_method')
        if detection_method != self.detection_method:
            self.detection_method = detection_method
            self._init_vis()

        if self.detection_method == 'groundtruth':
            impred = self.bridge.imgmsg_to_cv2(detectedge_response.prediction)
        if self.detection_method == 'network':
            corners = self.bridge.imgmsg_to_cv2(detectedge_response.corners)
            outer_edges = self.bridge.imgmsg_to_cv2(detectedge_response.outer_edges)
            inner_edges = self.bridge.imgmsg_to_cv2(detectedge_response.inner_edges)
            impred = np.zeros((corners.shape[0], corners.shape[1], 3), dtype=np.uint8)
            impred[:, :, 0] += corners
            impred[:, :, 1] += outer_edges
            impred[:, :, 2] += inner_edges
        elif self.detection_method == 'clothseg':
            impred = self.bridge.imgmsg_to_cv2(detectedge_response.prediction)
        elif self.detection_method == 'canny' or self.detection_method == 'canny_color':
            detection = self.bridge.imgmsg_to_cv2(detectedge_response.prediction)
            impred = detection[:, :, 0]
        elif self.detection_method == 'depthgrad':
            grads = self.bridge.imgmsg_to_cv2(detectedge_response.prediction)
            impred = grads[:, :, 0]
        elif self.detection_method == 'harris' or self.detection_method == 'harris_color':
            detection = self.bridge.imgmsg_to_cv2(detectedge_response.prediction)
            impred = detection[:, :, 0]
        return {
            'rgb_im': self.bridge.imgmsg_to_cv2(detectedge_response.rgb_im),
            'depth_im': self.bridge.imgmsg_to_cv2(detectedge_response.depth_im),
            'prediction': self.bridge.imgmsg_to_cv2(detectedge_response.prediction),
            'image_pred': impred,
            'corners': corners,
            'outer_edges': outer_edges,
            'inner_edges': inner_edges,
        }

    def _save(self, data, plot):
        tstamp = datetime.now().strftime("%d_%m_%Y_%H:%M:%S")
        dir_path = os.path.join(self.base_path, tstamp)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        np.save(os.path.join(dir_path, 'depth.npy'), data['depth_im'])
        rgb_im = cv2.cvtColor(data['rgb_im'], cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(dir_path, "rgb.png"), rgb_im)
        np.save(os.path.join(dir_path, 'pred.npy'), data['prediction'])

        if self.detection_method == 'network':
            impred = cv2.cvtColor(data['image_pred'], cv2.COLOR_BGR2RGB)
        else:
            impred = data['image_pred']
        cv2.imwrite(os.path.join(dir_path, 'impred.png'), impred)
        
        plot = cv2.cvtColor(plot, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(dir_path, 'plot.png'), plot)
        np.save(os.path.join(dir_path, 'data.npy'), data)

    def run(self):
        while not rospy.is_shutdown():
            detectedge_response = self._call_detectedge_service()
            data = self._aggregate_data(detectedge_response)
            plot = self.visualizer.visualize(data, show_grasp=False)

if __name__ == '__main__':
    cgs = ClothGrasper()
    cgs.run()
