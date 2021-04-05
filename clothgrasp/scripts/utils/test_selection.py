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
from clothgrasp.srv import DetectEdge, SelectGrasp, ProjectGrasp

class ClothGrasper:
    def __init__(self):
        rospy.init_node('clothgrasper')

        rospy.set_param('break_on_error', False)
        self.base_path = rospy.get_param('base_save_path')
        self.grasp_target = rospy.get_param('grasp_target')
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

    def _call_selectgrasp_service(self, detectedge_response):
        rospy.wait_for_service('select_grasp')
        select_grasp = rospy.ServiceProxy('select_grasp', SelectGrasp)
        rgb = detectedge_response.rgb_im
        prediction = detectedge_response.prediction
        corners = detectedge_response.corners
        outer_edges = detectedge_response.outer_edges
        inner_edges = detectedge_response.inner_edges
        return select_grasp(rgb, prediction, corners, outer_edges, inner_edges)

    def _aggregate_data(self, detectedge_response, selectgrasp_response):
        corners = None
        outer_edges = None
        inner_edges = None
        inner_py = None
        inner_px = None
        variance = None
        angle_x = None
        angle_y = None

        if self.detection_method == 'groundtruth':
            impred = self.bridge.imgmsg_to_cv2(detectedge_response.prediction)
            inner_py = selectgrasp_response.inner_py
            inner_px = selectgrasp_response.inner_px
            if len(selectgrasp_response.angle_x) != 0:
                shape = (impred.shape[0], impred.shape[1])
                variance = np.reshape(selectgrasp_response.var, shape)
                angle_x = np.reshape(selectgrasp_response.angle_x, shape)
                angle_y = np.reshape(selectgrasp_response.angle_y, shape)
        elif self.detection_method == 'network':
            corners = self.bridge.imgmsg_to_cv2(detectedge_response.corners)
            outer_edges = self.bridge.imgmsg_to_cv2(detectedge_response.outer_edges)
            inner_edges = self.bridge.imgmsg_to_cv2(detectedge_response.inner_edges)
            impred = np.zeros((corners.shape[0], corners.shape[1], 3), dtype=np.uint8)
            impred[:, :, 0] += corners
            impred[:, :, 1] += outer_edges
            impred[:, :, 2] += inner_edges
            inner_py = selectgrasp_response.inner_py
            inner_px = selectgrasp_response.inner_px
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
            'py': selectgrasp_response.py,
            'px': selectgrasp_response.px,
            'angle': selectgrasp_response.angle,
            'inner_py': inner_py,
            'inner_px': inner_px,
            'variance': variance,
            'angle_x': angle_x,
            'angle_y': angle_y
        }

    def _save_train(self, data, plot, count):
        if data['px'] == 0:
            return

        row_start, row_end, col_start, col_end, step = self.crop_dims

        pred = data['image_pred']
        red = pred[:, :, 0]
        yellow = pred[:, :, 1]
        green = pred[:, :, 2]
        fname = "%d_labels_%s.png"
        cv2.imwrite(os.path.join(self.base_path, fname % (count, "red")), red)
        cv2.imwrite(os.path.join(self.base_path, fname % (count, "yellow")), yellow)
        cv2.imwrite(os.path.join(self.base_path, fname % (count, "green")), green)
        
        rgb = data['rgb_im']
        rgb = rgb[row_start:row_end:step, col_start:col_end:step, :]
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        depth = data['depth_im']
        depth = depth[row_start:row_end:step, col_start:col_end:step]
        fname = "%s_%d.png"
        cv2.imwrite(os.path.join(self.base_path, fname % ("rgb", count)), rgb)
        fname = "%d_%s.npy"
        np.save(os.path.join(self.base_path, fname % (count, "depth")), depth)

        angle_x = data['angle_x']
        angle_y = data['angle_y']
        fname = "%d_direction_%s.npy"
        np.save(os.path.join(self.base_path, fname % (count, "x")), angle_x)
        np.save(os.path.join(self.base_path, fname % (count, "y")), angle_y)

        var = data['variance']
        fname = "%d_variance.npy"
        np.save(os.path.join(self.base_path, fname % count), var)

        invvar = 1/(var + 0.0001)
        fname = "%d_invvariance.npy"
        np.save(os.path.join(self.base_path, fname % count), invvar)

    def _save(self, data, plot):
        tstamp = datetime.now().strftime("%d_%m_%Y_%H:%M:%S:%f")
        dir_path = os.path.join(self.base_path, self.grasp_target, self.detection_method, tstamp)
        os.makedirs(dir_path)
        np.save(os.path.join(dir_path, 'depth.npy'), data['depth_im'])
        rgb_im = cv2.cvtColor(data['rgb_im'], cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(dir_path, "rgb.png"), rgb_im)
        np.save(os.path.join(dir_path, 'pred.npy'), data['prediction'])

        if self.detection_method == 'network' or self.detection_method == 'groundtruth':
            impred = cv2.cvtColor(data['image_pred'], cv2.COLOR_BGR2RGB)
        else:
            impred = data['image_pred']
        cv2.imwrite(os.path.join(dir_path, 'impred.png'), impred)
        
        plot = cv2.cvtColor(plot, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(dir_path, 'plot.png'), plot)
        np.save(os.path.join(dir_path, 'data.npy'), data)

    def run(self):
        count = 0
        while not rospy.is_shutdown():
            rospy.loginfo(count)
            detectedge_response = self._call_detectedge_service()
            selectgrasp_response = self._call_selectgrasp_service(detectedge_response)
            data = self._aggregate_data(detectedge_response, selectgrasp_response)
            plot = self.visualizer.visualize(data, show_grasp=True)
            # self._save(data, plot)
            # self._save_train(data, plot, count)
            rospy.sleep(0.3)
            count += 1

if __name__ == '__main__':
    cgs = ClothGrasper()
    cgs.run()
