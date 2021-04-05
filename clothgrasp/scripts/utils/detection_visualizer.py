import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from visualization_msgs.msg import Marker
from sensor_msgs.msg import Image
from pyquaternion import Quaternion as Quat
from copy import deepcopy

class DetectionVisualizer:
    def __init__(self, detection_method, crop_dims, arrow_scale=15):
        self.pub = rospy.Publisher('prediction', Image, queue_size=10)
        self.bridge = CvBridge()
        self.detection_method = detection_method
        self.crop_dims = crop_dims
        self.arrow_scale = arrow_scale

    def _network_output(self, data, show_grasp):
        rgb_im = data['rgb_im']
        impred = data['image_pred']
        if show_grasp:
            y = data['py']
            x = data['px']
            angle = data['angle']
            inner_y = data['inner_py']
            inner_x = data['inner_px']

        row_start, row_end, col_start, col_end, step = self.crop_dims
        rgb_im = rgb_im[row_start:row_end:step, col_start:col_end:step, :]
        
        plt.gcf().clear()
        fig = plt.figure()
        plt.imshow(rgb_im)
        plt.imshow(impred, alpha=0.7)

        if show_grasp and not (inner_y == y and inner_x == x):
            v = np.array([inner_x - x, inner_y - y])
            v = self.arrow_scale * (v / np.linalg.norm(v))
            v[1] *= -1 # opencv vs. matplotlib orientation conventions
            plt.quiver(x, y, v[0], v[1], color='red', scale_units='x', scale=1)

        fig.tight_layout()
        fig.canvas.draw()
        w,h = fig.canvas.get_width_height()
        buf = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
        return buf

    def _depth_output(self, data, show_grasp):
        depth = data['depth_im']
        rgb = data['rgb_im']
        impred = data['image_pred']

        if show_grasp:
            y = data['py']
            x = data['px']
            angle = data['angle']

        row_start, row_end, col_start, col_end, step = self.crop_dims
        depth = depth[row_start:row_end:step, col_start:col_end:step]
        rgb = rgb[row_start:row_end:step, col_start:col_end:step]

        plt.gcf().clear()
        fig = plt.figure()
        ax = plt.subplot(121)
        ax.imshow(impred, cmap='gray')
        if show_grasp:
            v = self.arrow_scale * np.array([np.cos(angle), np.sin(angle)])
            v[1] *= -1 # opencv vs. matplotlib orientation conventions
            plt.quiver(x, y, v[0], v[1], color='red', scale_units='x', scale=1)

        ax = plt.subplot(122)
        ax.imshow(rgb)
        ax.imshow(impred, alpha=0.8, cmap='gray')
        if show_grasp:
            v = self.arrow_scale * np.array([np.cos(angle), np.sin(angle)])
            v[1] *= -1 # opencv vs. matplotlib orientation conventions
            plt.quiver(x, y, v[0], v[1], color='red', scale_units='x', scale=1)

        fig.canvas.draw()
        w,h = fig.canvas.get_width_height()
        buf = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
        return buf

    def _rgb_output(self, data, show_grasp):
        depth = data['depth_im']
        rgb = data['rgb_im']
        impred = data['image_pred']
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        if show_grasp:
            y = data['py']
            x = data['px']
            angle = data['angle']

        row_start, row_end, col_start, col_end, step = self.crop_dims
        # depth = depth[row_start:row_end:step, col_start:col_end:step]
        rgb = rgb[row_start:row_end:step, col_start:col_end:step]

        plt.gcf().clear()
        fig = plt.figure()
        ax = plt.subplot(121)
        ax.imshow(impred, cmap='gray', vmin=0, vmax=.00001)
        if show_grasp:
            v = self.arrow_scale * np.array([np.cos(angle), np.sin(angle)])
            v[1] *= -1 # opencv vs. matplotlib orientation conventions
            plt.quiver(x, y, v[0], v[1], color='red', scale_units='x', scale=1)

        ax = plt.subplot(122)
        ax.imshow(rgb, cmap='gray')
        ax.imshow(impred, alpha=0.8, cmap='gray', vmin=0, vmax=.00001)
        if show_grasp:
            v = self.arrow_scale * np.array([np.cos(angle), np.sin(angle)])
            v[1] *= -1 # opencv vs. matplotlib orientation conventions
            plt.quiver(x, y, v[0], v[1], color='red', scale_units='x', scale=1)

        fig.canvas.draw()
        w,h = fig.canvas.get_width_height()
        buf = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
        return buf

    def visualize(self, data, show_grasp=True):
        """Plot and publish the data.
        """
        if self.detection_method == 'groundtruth':
            buf = self._network_output(data, show_grasp)
        elif self.detection_method == 'network':
            buf = self._network_output(data, show_grasp)
        elif self.detection_method == 'clothseg':
            buf = self._depth_output(data, show_grasp)
        elif self.detection_method == 'canny':
            buf = self._depth_output(data, show_grasp)
        elif self.detection_method == 'canny_color':
            buf = self._rgb_output(data, show_grasp)
        elif self.detection_method == 'harris':
            buf = self._depth_output(data, show_grasp)
        elif self.detection_method == 'harris_color':
            buf = self._rgb_output(data, show_grasp)
        else:
            raise NotImplementedError

        msg = self.bridge.cv2_to_imgmsg(buf, encoding='rgb8')
        self.pub.publish(msg)
        return buf

