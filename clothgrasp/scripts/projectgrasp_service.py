#!/usr/bin/env python
import os
import rospy
import cv2
import tf
import numpy as np
from clothgrasp.srv import ProjectGrasp, ProjectGraspResponse
from geometry_msgs.msg import Pose
from cv_bridge import CvBridge
from pyquaternion import Quaternion as Quat
from tf.transformations import quaternion_from_matrix
from copy import deepcopy

class GraspPlanner():
    def __init__(self):
        rospy.init_node('projectgrasp_service')

        self.bridge = CvBridge()
        self.pose_world_camera = rospy.get_param('w2c_pose')
        self.T_world_camera = self._get_tf_from_arr(self.pose_world_camera)

        self.detection_method = rospy.get_param('detection_method')
        self.crop_dims = rospy.get_param('crop_dims') if self.detection_method == 'network' or self.detection_method == 'groundtruth' else rospy.get_param('crop_dims_baselines')

        self.D = np.array(rospy.get_param('D'))
        self.K = np.array(rospy.get_param('K'))
        self.K = np.reshape(self.K, (3, 3))

        self.server = rospy.Service('project_grasp', ProjectGrasp, self._server_cb)

    def _get_tf_from_arr(self, arr):
        q = Quat(x=arr[3], y=arr[4], z=arr[5], w=arr[6])
        tf_mat = q.transformation_matrix
        tf_mat[0:3, 3] = arr[0:3]
        return tf_mat

    def _get_pose_from_tf(self, t):
        pose = Pose()
        pose.position.x = t[0, -1]
        pose.position.y = t[1, -1]
        pose.position.z = t[2, -1]
        q = quaternion_from_matrix(t)
        pose.orientation.x = q[0]
        pose.orientation.y = q[1]
        pose.orientation.z = q[2]
        pose.orientation.w = q[3]
        return pose

    def _pose_vector(self, pose):
        """Returns unit pose vector.
        """
        q = Quat(w=pose.orientation.w, x=pose.orientation.x, y=pose.orientation.y, z=pose.orientation.z).unit
        unit_v = np.array([0., 0., 1.]) # rotate world frame z axis
        pose_v = q.rotate(unit_v)
        return pose_v

    def _get_offset_pose(self, pose, offset=0.1):
        """Returns translation offset from pose.
        """
        # Get pose vector to compute offset
        v = self._pose_vector(pose)
        offset_pose = deepcopy(pose)
        offset_pose.position.x -= offset*v[0]
        offset_pose.position.y -= offset*v[1]
        offset_pose.position.z -= offset*v[2]
        return offset_pose

    def _transform_grasp(self, depth_im, grasp_pt, angle):
        y, x = grasp_pt
        y *= self.crop_dims[-1]
        x *= self.crop_dims[-1]
        y += self.crop_dims[0] # account for cropping and scaling
        x += self.crop_dims[2]

        # Transform grasp point from camera to world frame
        f_x = self.K[0, 0]
        f_y = self.K[1, 1]
        c_x = self.K[0, 2]
        c_y = self.K[1, 2]

        # Undistort points
        points = np.array([[x, y]], dtype=np.float32)
        points_undistorted = cv2.undistortPoints(np.expand_dims(points, axis=0), self.K, self.D, P=self.K)
        points_undistorted = np.squeeze(points_undistorted, axis=0)

        z = depth_im[y][x]
        px = (points_undistorted[0, 0] - c_x) / f_x * z
        py = (points_undistorted[0, 1] - c_y) / f_y * z

        # Transform grasp in camera frame to world frame
        T_camera_grasp = self._get_tf_from_arr([px, py, z, 0, 0, 0, 1])

        T_world_grasp = self.T_world_camera.dot(T_camera_grasp)
        pose_world_grasp = self._get_pose_from_tf(T_world_grasp)

        # Set orientation to top down
        pose_world_grasp.orientation.x = 1
        pose_world_grasp.orientation.y = 0
        pose_world_grasp.orientation.z = 0
        pose_world_grasp.orientation.w = 0
        
        pose_q = Quat(x=pose_world_grasp.orientation.x, y=pose_world_grasp.orientation.y, z=pose_world_grasp.orientation.z, w=pose_world_grasp.orientation.w)
        
        angle_q = Quat(axis=(0, 0, 1), angle=angle)
        q = pose_q * angle_q
        pose_world_grasp.orientation.x = q[1]
        pose_world_grasp.orientation.y = q[2]
        pose_world_grasp.orientation.z = q[3]
        pose_world_grasp.orientation.w = q[0]
        return pose_world_grasp

    def _server_cb(self, req):
        rospy.loginfo('Received grasp projection request')
        depth_im = deepcopy(self.bridge.imgmsg_to_cv2(req.depth_im))
        grasp_pt = [req.py, req.px]
        angle = req.angle

        response = ProjectGraspResponse()
        cloth_pose = self._transform_grasp(depth_im, grasp_pt, angle)
        response.cloth_pose = cloth_pose
        rospy.loginfo('Sending grasp projection response')
        return response

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    g = GraspPlanner()
    g.run()
