#! /usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import rospy
import moveit_commander
import moveit_msgs.msg
from geometry_msgs.msg import PoseStamped
import intera_interface
import numpy as np
import tf.transformations as trans

xdiff = 0.0593
base_standoff = 0.14
sensor_standoff_z = 0.1
sensor_standoff_y = 0.2

def initScene(scene, table_height, table_shape=[0.7, 1.6]):
    scene.remove_world_object("table")
    scene.remove_world_object("sensor1")
    scene.remove_world_object("sensor1_rod")
    scene.remove_world_object("wall_br")
    scene.remove_world_object("wall_left")
    scene.remove_world_object("wall_right") 
    scene.remove_world_object("workspace")

    time_stamp = rospy.get_rostime()
    
    table_center = [base_standoff, 0.0, table_height]
    table_pose = PoseStamped()
    table_pose.header.frame_id = 'base'
    table_pose.header.stamp = time_stamp
    table_pose.pose.orientation.w = 1.0
    table_pose.pose.position.x = table_center[0] + xdiff
    table_pose.pose.position.y = table_center[1]
    table_pose.pose.position.z = table_center[2]
    table_size = tuple(table_shape) + (0.0, )

    sensor1_size = [0.25, 0.20, 0.1875]
    sensor1_pose = PoseStamped()
    sensor1_pose.header.frame_id = 'base'
    sensor1_pose.header.stamp = time_stamp
    sensor1_pose.pose.orientation.w = 1.0
    sensor1_pose.pose.position.x = 0.7398
    sensor1_pose.pose.position.y = -0.2510
    sensor1_pose.pose.position.z = 0.6639 + 0.015
    
    sensor1_arm_size = [0.15, 0.23, 0.15]
    sensor1_arm_pose = PoseStamped()
    sensor1_arm_pose.header.frame_id = 'base'
    sensor1_arm_pose.header.stamp = time_stamp
    sensor1_arm_pose.pose.orientation.w = 1.0
    sensor1_arm_pose.pose.position.x = 0.7398
    sensor1_arm_pose.pose.position.y = -0.2510 - 0.19
    sensor1_arm_pose.pose.position.z = 0.6639

    sensor1_rod_height = 0.6639 + 0.015
    sensor1_rod_size = [0.06, 0.06, sensor1_rod_height]
    sensor1_rod_pose = PoseStamped()
    sensor1_rod_pose.header.frame_id = 'base'
    sensor1_rod_pose.header.stamp = time_stamp
    sensor1_rod_pose.pose.orientation.w = 1.0
    sensor1_rod_pose.pose.position.x = 0.7398 - 0.03
    sensor1_rod_pose.pose.position.y = -0.2510 - 0.19-0.04 - 0.01
    sensor1_rod_pose.pose.position.z = sensor1_rod_height / 2

    wall_height = 1.5
    wall_b_size = (0.0, 2, wall_height)
    wall_br_pose = PoseStamped()
    wall_br_pose.header.frame_id = 'base'
    wall_br_pose.header.stamp = time_stamp
    wall_br_pose.pose.orientation.w = 1
    wall_br_pose.pose.position.x = -0.35
    wall_br_pose.pose.position.y = 0
    wall_br_pose.pose.position.z = wall_height/2 + table_center[2]

    wall_height = 1.5
    wall_f_size = (0.0, 2, wall_height)
    wall_f_pose = PoseStamped()
    wall_f_pose.header.frame_id = 'base'
    wall_f_pose.header.stamp = time_stamp
    wall_f_pose.pose.orientation.w = 1
    wall_f_pose.pose.position.x = 1.0 + xdiff
    wall_f_pose.pose.position.y = 0
    wall_f_pose.pose.position.z = wall_height/2 + table_center[2]
    
    wall_height = 1.5
    wall_r_size = (2, 0.0, wall_height)
    wall_r_pose = PoseStamped()
    wall_r_pose.header.frame_id = 'base'
    wall_r_pose.header.stamp = time_stamp
    wall_r_pose.pose.orientation.w = 1.0
    wall_r_pose.pose.position.x = 0.0 + xdiff
    wall_r_pose.pose.position.y = -1.0
    wall_r_pose.pose.position.z = wall_height/2 + table_center[2]

    wall_height = 1.5
    wall_l_size = (2, 0.0, wall_height)
    wall_l_pose = PoseStamped()
    wall_l_pose.header.frame_id = 'base'
    wall_l_pose.header.stamp = time_stamp
    wall_l_pose.pose.orientation.w = 1.0
    wall_l_pose.pose.position.x = 0.0 + xdiff
    wall_l_pose.pose.position.y = 1.0
    wall_l_pose.pose.position.z = wall_height/2 + table_center[2]

    sleep = 0.3
    rospy.sleep(sleep)
    scene.add_box('wall_br', wall_br_pose, wall_b_size)
    rospy.sleep(sleep)
    scene.add_box('wall_right', wall_r_pose, wall_r_size)
    rospy.sleep(sleep)
    scene.add_box('wall_left', wall_l_pose, wall_l_size)
    rospy.sleep(sleep)
    scene.add_box('wall_front', wall_f_pose, wall_f_size)
    rospy.sleep(sleep)
    scene.add_box('table', table_pose, table_size)
    rospy.sleep(sleep)
    scene.add_box('sensor1', sensor1_pose, sensor1_size)
    rospy.sleep(sleep)
    scene.add_box('sensor1_arm', sensor1_arm_pose, sensor1_arm_size)
    rospy.sleep(sleep)
    scene.add_box('sensor1_rod', sensor1_rod_pose, sensor1_rod_size)
    rospy.sleep(sleep)

try:
    rospy.init_node("scene_geometry")
    scene = moveit_commander.PlanningSceneInterface()
    
    initScene(scene, 
                table_height=-0.02,
                table_shape=[2.0, 2])
except rospy.ROSInterruptException:
    pass
