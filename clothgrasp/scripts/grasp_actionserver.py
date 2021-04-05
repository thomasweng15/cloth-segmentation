#!/usr/bin/env python
import sys
import cv2
import rospy
import numpy as np
import actionlib
import moveit_commander
import intera_interface
from copy import deepcopy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Pose, PoseStamped
from moveit_msgs.msg import PlanningOptions, RobotState
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from pyquaternion import Quaternion as Quat
from utils.marker_visualizer import MarkerVisualizer
from utils.utils import unit_vector_from_pose
import tf

from clothgrasp.msg import ExecuteGraspAction, ExecuteGraspResult, MoveHomeAction, MoveHomeResult
from wsg_50_common.srv import Move
from wsg_50_common.msg import Status

class Sawyer():
    def __init__(self):
        # Get parameters
        self.init_joint_angles = rospy.get_param('init_joint_angles')
        self.pregrasp_joint_angles = rospy.get_param('pregrasp_joint_angles')
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.group = moveit_commander.MoveGroupCommander('right_arm')
        self.max_vel = 0.4
        self.group.set_max_velocity_scaling_factor(self.max_vel)
        self.group.set_max_acceleration_scaling_factor(0.4)
        self.group.set_planning_time(2.0) # 5 is default
        self.group.set_num_planning_attempts(20)
        self.group.set_planner_id("RRTstarkConfigDefault")
        self.sim = rospy.get_param('sim')

        self.listener = tf.TransformListener()
        self.viz = MarkerVisualizer()

        self.gripper_pos = 0
        self.gripper_ready = False
        if not self.sim:
            rospy.wait_for_service('/wsg_50_driver/move')
            self.move = rospy.ServiceProxy('/wsg_50_driver/move', Move)
            self.gripper_status_sub = rospy.Subscriber('/wsg_50_driver/status', Status, self._gripper_status_cb, queue_size=1)

        self.xy_slide_offset = 0.02
        self.slide_multiplier = 2.0

    def _gripper_status_cb(self, msg):
        self.gripper_pos = msg.width
        self.gripper_ready = "Fast Stop" not in msg.status

    def move_to_angles(self, goal):
        plan = self.group.go(goal, wait=True)
        if plan is False:
            return False
        self.group.stop()
        return True
    
    def move_home(self):
        return self.move_to_angles(self.init_joint_angles)

    def lift(self):
        return self.move_to_angles(self.pregrasp_joint_angles)

    def open_gripper(self):
        if self.sim: 
            return True
        if not self.gripper_ready:
            rospy.logerr('Gripper not ready')
            raise Exception
        resp = self.move(60, 40)
        rospy.loginfo(resp)
        return True

    def close_gripper(self):
        if self.sim:
            return True 
        if not self.gripper_ready:
            rospy.logerr('Gripper not ready')
            raise Exception
        resp = self.move(0, 40)
        rospy.loginfo(resp)
        return True

    def _get_slide_poses(self, cloth_pose):
        v = unit_vector_from_pose(cloth_pose)
        pose = deepcopy(cloth_pose)

        # Transform to tip of end effector
        if rospy.get_param('transform_gripper'):
            self.listener.waitForTransform("right_hand", "right_gripper_tip", rospy.Time(0), rospy.Duration(4.0))
            t, _ = self.listener.lookupTransform("right_hand", "right_gripper_tip", rospy.Time(0))
            pose.position.x -= t[2]*v[0]
            pose.position.y -= t[2]*v[1]
            pose.position.z -= t[2]*v[2]

        # Offset pose to get slide start pose and slide end pose
        start_pose = deepcopy(pose)
        end_pose = deepcopy(pose)
        start_pose.position.x -= self.xy_slide_offset*v[0]
        start_pose.position.y -= self.xy_slide_offset*v[1]
        end_pose.position.x += self.slide_multiplier*self.xy_slide_offset*v[0]
        end_pose.position.y += self.slide_multiplier*self.xy_slide_offset*v[1]

        return start_pose, end_pose

    def _get_approach_pose(self, pose):
        approach_offset = 0.02
        v = unit_vector_from_pose(pose)
        offset_pose = deepcopy(pose)
        offset_pose.position.x -= approach_offset*v[0]
        offset_pose.position.y -= approach_offset*v[1]
        offset_pose.position.z -= approach_offset*v[2]
        return offset_pose

    def add_workspace_collision(self):
        """Add workspace collision box to prevent planning through cloth.
        """
        self.scene.remove_world_object("workspace")
        time_stamp = rospy.get_rostime()
        ws_height = 0.02
        ws_size = [0.6, 1.3, ws_height]
        ws_pose = PoseStamped()
        ws_pose.header.frame_id = 'base'
        ws_pose.header.stamp = time_stamp
        ws_pose.pose.orientation.w = 1.0
        ws_pose.pose.position.x = 0.55
        ws_pose.pose.position.y = 0.0
        ws_pose.pose.position.z = -0.05 + ws_height/2
        self.scene.add_box('workspace', ws_pose, ws_size)
        rospy.sleep(0.5)

    def _get_approach_plan(self, slide_start_pose):
        approach_pose = self._get_approach_pose(slide_start_pose)

        rospy.loginfo("Plan to approach pose...")
        self.add_workspace_collision()
        self.group.set_start_state_to_current_state()
        self.group.set_pose_target(approach_pose)
        approach_plan = self.group.plan()
        self.scene.remove_world_object("workspace")
        if len(approach_plan.joint_trajectory.points) == 0:
            rospy.logerr("Could not plan to approach pose.")
            return 0, None, None

        rospy.loginfo("Plan to slide start pose...")
        start_point = approach_plan.joint_trajectory.points[-1]
        state = RobotState()
        state.joint_state.name = approach_plan.joint_trajectory.joint_names
        state.joint_state.position = start_point.positions
        state.joint_state.velocity = start_point.velocities
        state.joint_state.effort = start_point.effort
        self.group.set_start_state(state)
        self.group.set_pose_target(slide_start_pose)
        slide_start_plan = self.group.plan()
        if len(slide_start_plan.joint_trajectory.points) == 0:
            rospy.logerr("Could not plan to slide start pose.")
            return 0, None, None

        self.viz.set_marker(approach_pose, [1, 0, 0, 0.7], 0)
        self.viz.set_marker(slide_start_pose, [0, 1, 0, 0.7], 1)

        plan_length = len(approach_plan.joint_trajectory.points) + \
                      len(slide_start_plan.joint_trajectory.points)
        return plan_length, approach_plan, slide_start_plan

    def plan(self, goal):
        """Plan for both grasp pose and rotated grasp pose 
        """
        cloth_pose = goal.cloth_pose

        rospy.logwarn('Manually overriding cloth pose')
        cloth_pose.position.x =  0.631810528243
        cloth_pose.position.y = -0.0675894045423
        cloth_pose.position.z = 0.037017763171
        cloth_pose.orientation.x =  0.653281482438
        cloth_pose.orientation.y =  0.653281482438
        cloth_pose.orientation.z =  -0.270598050073
        cloth_pose.orientation.w =  -0.270598050073

        rospy.loginfo("Get plan for default cloth pose")
        slide_start_pose, slide_end_pose = self._get_slide_poses(cloth_pose)
        self.viz.set_marker(cloth_pose, [0, 0, 1, 0.7], 2)
        self.viz.set_marker(slide_end_pose, [0, 1, 1, 0.7], 3)
        plan_length, approach_plan, slide_start_plan = self._get_approach_plan(slide_start_pose)

        if plan_length == 0:
            rospy.logerr("Could not plan a complete trajectory.")
            return None, None, None
        return approach_plan, slide_start_pose, slide_end_pose

    def execute_plan(self, plan):
        self.group.set_start_state_to_current_state()
        if not self.group.execute(plan, wait=True):
            return False
        self.group.stop()
        self.group.clear_pose_targets()
        return True

    def move_to_pose(self, pose):
        self.group.set_start_state_to_current_state()
        self.group.set_pose_target(pose)
        plan = self.group.plan()
        if len(plan.joint_trajectory.points) == 0:
            rospy.logerr("Failed to plan to slide start pose")
            return False
        return self.execute_plan(plan)

    def get_waypoints(self, curr_pose, goal_pose):
        waypoints = []
        diff_x = goal_pose.position.x - curr_pose.position.x
        diff_y = goal_pose.position.y - curr_pose.position.y
        diff_z = goal_pose.position.z - curr_pose.position.z
        for i in range(5):
            curr_pose.position.x += diff_x / 5.0
            curr_pose.position.y += diff_y / 5.0
            curr_pose.position.z += diff_z / 5.0
        waypoints.append(deepcopy(curr_pose))
        return waypoints

    def cartesian(self, start_pose, end_pose):
        waypoints = self.get_waypoints(start_pose, end_pose)
        (plan, fraction) = self.group.compute_cartesian_path(
                                waypoints,   # waypoints to follow
                                0.01,        # eef_step
                                5.00)        # jump_threshold
        
        if fraction < 1.0:
            return False

        if not self.group.execute(plan, wait=True):
            return False

    def publish_viz(self):
        self.viz.publish()

class GraspActionServer():
    def __init__(self):
        rospy.init_node('grasp_actionserver')
        moveit_commander.roscpp_initialize(['joint_states:=/robot/joint_states'])
        self.robot = Sawyer()
        self.bridge = CvBridge()
        self.rgb_sub = rospy.Subscriber('/rgb/image_raw', Image, self._rgb_cb, queue_size=1)
        self.move_server = actionlib.SimpleActionServer('move_home', MoveHomeAction, self._move_home, False)
        self.move_server.start()
        self.grasp_server = actionlib.SimpleActionServer('execute_grasp', ExecuteGraspAction, self.execute, False)
        self.grasp_server.start()
        self.grasp_result = ExecuteGraspResult()
    
    def _rgb_cb(self, msg):
        self.rgb = self.bridge.imgmsg_to_cv2(msg)

    def _move_home(self, goal):
        rospy.loginfo('Received move home request...')
        move_result = MoveHomeResult()
        if self.robot.move_home():
            rospy.loginfo('Move home succeeded.')
            self.move_server.set_succeeded(move_result)
        else:
            rospy.loginfo('Move home failed.')
            self.move_server.set_aborted()

    def execute(self, goal):
        rospy.loginfo('Received grasp request...')

        self.robot.open_gripper()

        rospy.loginfo("Plan trajectory to approach pose")
        approach_plan, slide_start_pose, slide_end_pose = self.robot.plan(goal)
        if approach_plan == None:
            self.grasp_server.set_aborted()
            return

        rospy.loginfo("Move to approach pose")
        if not self.robot.execute_plan(approach_plan):
            rospy.logerr("Failed to reach approach pose")
            self.grasp_server.set_aborted()
            return

        rospy.loginfo("Move to slide start pose")
        if not self.robot.move_to_pose(slide_start_pose):
            self.grasp_server.set_aborted()
            return

        rospy.loginfo("Execute slide")
        if not self.robot.cartesian(slide_start_pose, slide_end_pose):
            self.grasp_server.set_aborted()
            return

        self.robot.close_gripper()

        rospy.loginfo("Move to lift pose")
        self.robot.lift()

        rospy.loginfo("Grasp succeeded.")
        grasp_result = ExecuteGraspResult()
        self.grasp_server.set_succeeded(grasp_result)

    def run(self):
        while not rospy.is_shutdown():
            self.robot.publish_viz()
            rospy.sleep(0.01)

if __name__ == '__main__':
    g = GraspActionServer()
    g.run()
