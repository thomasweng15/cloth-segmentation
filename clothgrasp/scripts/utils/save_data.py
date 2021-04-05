
#!/usr/bin/env python
import os
import rospy
import sys
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import tf
import time
import numpy as np
from copy import deepcopy
import datetime
import message_filters
from sensor_msgs.msg import Image, CameraInfo

class Capture():
    def __init__(self):
        self.filepath = sys.argv[1]
        self.bridge = CvBridge()

        rospy.init_node('listener', anonymous=True)

        self.id = 0

        self.image_sub = message_filters.Subscriber('/rgb/image_raw', Image)
        self.depth_sub = message_filters.Subscriber('/depth_to_rgb/image_raw', Image)

        if not os.path.exists(self.filepath):
            os.mkdir(self.filepath)
        self.ts = message_filters.ApproximateTimeSynchronizer([self.image_sub, self.depth_sub], 1000,10)

        self.ts.registerCallback(self.callback)
        rospy.spin()

    def callback(self, image0, depth0):
        rospy.loginfo(self.id)
        img_path = os.path.join(self.filepath, 'rgb_%d_%s-%s.png' % (self.id, image0.header.stamp.secs, image0.header.stamp.nsecs))
        rgb_im = self.bridge.imgmsg_to_cv2(image0, desired_encoding="passthrough")
        cv2.imwrite(img_path, rgb_im)
        img_path = os.path.join(self.filepath, 'depth_%d_%s-%s.npy' % (self.id, depth0.header.stamp.secs, depth0.header.stamp.nsecs))
        depth_im = self.bridge.imgmsg_to_cv2(depth0, desired_encoding="passthrough")
        np.save(img_path, depth_im)

        print(img_path)
        self.id += 1
        rospy.shutdown()


if __name__ == "__main__":
    c = Capture()






