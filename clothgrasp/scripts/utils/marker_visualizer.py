import rospy
import numpy as np
from visualization_msgs.msg import Marker
from pyquaternion import Quaternion as Quat
from copy import deepcopy

class MarkerVisualizer:
    def __init__(self):
        self.pub = rospy.Publisher('visualization_marker', Marker, queue_size=10) 
        self.markers = {}

    def _rotate_marker_pose(self, pose):
        """Rotate pose so it points down z axis instead of x axis.
        """
        q = Quat(w=pose.orientation.w,
                 x=pose.orientation.x,
                 y=pose.orientation.y,
                 z=pose.orientation.z)
        axis_rot = Quat(axis=[0., 1., 0.], degrees=-90)
        rotq = q * axis_rot
        rotpose = deepcopy(pose)
        rotpose.orientation.w = rotq[0]
        rotpose.orientation.x = rotq[1]
        rotpose.orientation.y = rotq[2]
        rotpose.orientation.z = rotq[3]
        return rotpose

    def set_marker(self, pose, colors, marker_id, frame='world', shape=0):
        """Create visualization marker for pose.
        """
        marker = Marker()
        marker.header.frame_id = frame
        marker.header.stamp = rospy.get_rostime()
        marker.ns = ""
        marker.id = marker_id
        marker.type = shape # arrow
        marker.action = 0 # add/modify
        marker.pose = self._rotate_marker_pose(pose)
        if shape == 2:
            marker.scale.x = 0.02
            marker.scale.y = 0.02
            marker.scale.z = 0.02
        else:
            marker.scale.x = 0.1
            marker.scale.y = 0.02
            marker.scale.z = 0.02

        marker.color.r = colors[0]
        marker.color.g = colors[1]
        marker.color.b = colors[2]
        marker.color.a = colors[3]

        marker.lifetime = rospy.Duration(6000) # no timeout
        
        self.markers[marker_id] = marker

    def publish(self):
        for key in self.markers.keys():
            self.pub.publish(self.markers[key])

    def clear(self):
        for key in self.markers.keys():
            marker = self.markers[key]
            marker.action = 2
            marker.lifetime = rospy.Duration(0.1)
            self.pub.publish(marker)