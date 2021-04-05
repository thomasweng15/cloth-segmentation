#!/usr/bin/env python

import rospy
import intera_interface

rospy.init_node("init_arm_sim")
limb = intera_interface.Limb('right')
starting_joint_angles = {'right_j0': 0.4236376953125,
                        'right_j1': -0.843982421875,
                        'right_j2': -1.57226171875,
                        'right_j3': 1.4317373046875,
                        'right_j4': 0.7826015625,
                        'right_j5': 1.661955078125,
                        'right_j6': 3.80976953125}

limb.move_to_joint_positions(starting_joint_angles)
