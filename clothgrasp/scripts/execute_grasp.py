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
from geometry_msgs.msg import Pose
from utils.detection_visualizer import DetectionVisualizer
from clothgrasp.msg import ExecuteGraspAction, ExecuteGraspGoal, MoveHomeAction, MoveHomeGoal
from clothgrasp.srv import DetectEdge, SelectGrasp, ProjectGrasp

class ClothGrasper:
    """
    Runs the pipeline to execute a sliding grasp on a cloth.
    """
    def __init__(self):
        rospy.init_node('clothgrasper')

        rospy.set_param('break_on_error', True)
        self.base_path = rospy.get_param('base_save_path')
        self.grasp_target = rospy.get_param('grasp_target')
        self.detection_method = rospy.get_param('detection_method')
        self.crop_dims = rospy.get_param('crop_dims') if self.detection_method == 'network' or self.detection_method == 'groundtruth' else rospy.get_param('crop_dims_baselines')

        self.move_client = actionlib.SimpleActionClient('move_home', MoveHomeAction)
        self.move_client.wait_for_server()
        self.grasp_client = actionlib.SimpleActionClient('execute_grasp', ExecuteGraspAction)
        self.grasp_client.wait_for_server()
        self.bridge = CvBridge()
        self.visualizer = DetectionVisualizer(self.detection_method, self.crop_dims)

    def _move_home(self):
        goal = MoveHomeGoal()
        self.move_client.send_goal(goal)
        self.move_client.wait_for_result()
        return self.move_client.get_result()

    def _call_detectedge_service(self):
        start = rospy.Time.now()
        rospy.wait_for_service('detect_edges')
        detect_edge = rospy.ServiceProxy('detect_edges', DetectEdge)
        response = detect_edge()
        d = rospy.Time.now() - start
        rospy.loginfo('Detect secs: %d, nsecs: %d' % (d.secs, d.nsecs))
        return response

    def _call_selectgrasp_service(self, detectedge_response):
        start = rospy.Time.now()
        rospy.wait_for_service('select_grasp')
        select_grasp = rospy.ServiceProxy('select_grasp', SelectGrasp)
        rgb = detectedge_response.rgb_im # for debugging
        prediction = detectedge_response.prediction
        corners = detectedge_response.corners
        outer_edges = detectedge_response.outer_edges
        inner_edges = detectedge_response.inner_edges
        response = select_grasp(rgb, prediction, corners, outer_edges, inner_edges)
        d = rospy.Time.now() - start
        rospy.loginfo('Select Grasp secs: %d, nsecs: %d' % (d.secs, d.nsecs))
        return response

    def _call_projectgrasp_service(self, detectedge_response, selectgrasp_response):
        start = rospy.Time.now()
        rospy.wait_for_service('project_grasp')
        project_grasp = rospy.ServiceProxy('project_grasp', ProjectGrasp)
        depth_im = detectedge_response.depth_im
        py = selectgrasp_response.py
        px = selectgrasp_response.px
        angle = selectgrasp_response.angle
        response = project_grasp(depth_im, py, px, angle)
        d = rospy.Time.now() - start
        rospy.loginfo('Plan Grasp secs: %d, nsecs: %d' % (d.secs, d.nsecs))
        return response

    def _execute_grasp(self, projectgrasp_response):
        start = rospy.Time.now()
        cloth_pose = projectgrasp_response.cloth_pose
        # cloth_pose = Pose() # debug
        goal = ExecuteGraspGoal(cloth_pose)
        self.grasp_client.send_goal(goal)
        self.grasp_client.wait_for_result()
        response = self.grasp_client.get_result()
        d = rospy.Time.now() - start
        rospy.loginfo('Execute Grasp secs: %d, nsecs: %d' % (d.secs, d.nsecs))
        return response

    def _aggregate_data(self, detectedge_response, selectgrasp_response, projectgrasp_response):
        corners = None
        outer_edges = None
        inner_edges = None
        inner_py = None
        inner_px = None

        if self.detection_method == 'groundtruth':
            impred = self.bridge.imgmsg_to_cv2(detectedge_response.prediction)
            inner_py = selectgrasp_response.inner_py
            inner_px = selectgrasp_response.inner_px
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
            grads = self.bridge.imgmsg_to_cv2(detectedge_response.prediction)
            impred = grads[:, :, 0]
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
            'cloth_pose': projectgrasp_response.cloth_pose
        }

    def _save(self, data, plot, state, grasp_result):
        tstamp = datetime.now().strftime("%d_%m_%Y_%H:%M:%S")
        if state == 3:
            dir_path = os.path.join(self.base_path, self.grasp_target, self.detection_method, tstamp)
        else: 
            dir_path = os.path.join(self.base_path, self.grasp_target, self.detection_method, "planning_failures", tstamp)

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
        self._move_home()

        detectedge_response = self._call_detectedge_service()
        selectgrasp_response = self._call_selectgrasp_service(detectedge_response)
        if selectgrasp_response.px == 0 and selectgrasp_response.py == 0:
            rospy.logerr("No cloth detected")
            return False

        projectgrasp_response = self._call_projectgrasp_service(detectedge_response, selectgrasp_response)

        data = self._aggregate_data(detectedge_response, selectgrasp_response, projectgrasp_response)
        plot = self.visualizer.visualize(data)

        # projectgrasp_response = None # debug
        grasp_result = self._execute_grasp(projectgrasp_response)
        state = self.grasp_client.get_state()
        self._save(data, plot, state, grasp_result)

        self._move_home()

if __name__ == '__main__':
    cgs = ClothGrasper()
    cgs.run()