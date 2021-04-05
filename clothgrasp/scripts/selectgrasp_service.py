#!/usr/bin/env python
import matplotlib
import matplotlib.pyplot as plt
import os
import rospy
import cv2
import message_filters
import numpy as np
from clothgrasp.srv import SelectGrasp, SelectGraspResponse
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose
from cv_bridge import CvBridge
from pyquaternion import Quaternion as Quat
from tf.transformations import quaternion_from_matrix
from visualization_msgs.msg import Marker
from copy import deepcopy
from datetime import datetime
from sklearn.neighbors import KDTree

class GroundTruthSelector:
    """Grasp selector using confidence prediction and ground truth labels
    """
    def __init__(self):
        self.bridge = CvBridge()
        self.pub = rospy.Publisher('pheatmap', Image, queue_size=1)

    def select_grasp(self, impred):
        segmentation = deepcopy(impred)
        segmentation[np.logical_and(impred[:,:,1]==255, impred[:,:,2]==255),2] = 0
        segmentation[np.logical_and(impred[:,:,1]==255, impred[:,:,2]==255),1] = 0
        im_height, im_width, _ = segmentation.shape
        
        # Get outer-inner edge correspondence
        xx, yy =  np.meshgrid([x for x in range(im_width)],
                              [y for y in range(im_height)])
        grasp_target = rospy.get_param('grasp_target')
        if grasp_target == 'edges':
            xx_o = xx[segmentation[:,:,1]==255]
            yy_o = yy[segmentation[:,:,1]==255]
        else:
            xx_o = xx[segmentation[:,:,0]==255]
            yy_o = yy[segmentation[:,:,0]==255]

        xx_i = xx[segmentation[:,:,2]==255]
        yy_i = yy[segmentation[:,:,2]==255]

        inner_edges_mask = np.ones((im_height, im_width))
        inner_edges_mask[segmentation[:,:,2]==255] = 0
        dists, lbl = cv2.distanceTransformWithLabels(inner_edges_mask.astype(np.uint8), cv2.DIST_L2, 
                                                    5, labelType=cv2.DIST_LABEL_PIXEL)

        # lbl provides at each pixel the index label of the closest zero point. 
        # Now we find the pixel coordinate of that index label
        inner_pxs = np.where(inner_edges_mask==0)
        xx_inner = inner_pxs[1] # x coords of inner edges
        yy_inner = inner_pxs[0] # y coords of inner edges
        labels_to_pxs = [[0, 0]] # fix off by one offset
        for j in range(len(yy_inner)):
            labels_to_pxs.append([yy_inner[j],xx_inner[j]])
        labels_to_pxs = np.array(labels_to_pxs)
        closest_inner_px = labels_to_pxs[lbl]
        
        # Calculate distance to the closest inner edge point for every pixel in the image
        dist_to_inner = np.zeros(closest_inner_px.shape)
        dist_to_inner[:,:,0] = np.abs(closest_inner_px[:,:,0]-yy)
        dist_to_inner[:,:,1] = np.abs(closest_inner_px[:,:,1]-xx)
        
        # Normalize distance vectors
        mag = np.linalg.norm([dist_to_inner[:,:,0],dist_to_inner[:,:,1]],axis = 0)+0.00001
        dist_to_inner[:,:,0] = dist_to_inner[:,:,0]/mag
        dist_to_inner[:,:,1] = dist_to_inner[:,:,1]/mag

        # For every outer edge point, find its closest K neighbours 
        num_neighbour = 100
        outer_idxs = np.vstack([xx_o,yy_o])
        try:
            tree = KDTree(outer_idxs.T, leaf_size=2)
        except Exception as e:
            print(e)
            return 0, 0, 0, 0, 0, None, None, None
        if num_neighbour > xx_o.shape[0]:
            print("Error: Num neighbors larger than number of outer edges")
            return 0, 0, 0, 0, 0, None, None, None
        dist, ind = tree.query(outer_idxs.T, k=num_neighbour)

        dist_to_inner_o = dist_to_inner[segmentation[:,:,1]==255,:]
        xx_neighbours = dist_to_inner_o[ind][:,:,1]
        yy_neighbours = dist_to_inner_o[ind][:,:,0]
        xx_var = np.var(xx_neighbours,axis = 1)
        yy_var = np.var(yy_neighbours,axis = 1)
        var = xx_var+yy_var

        var_max = var.max()
        var = var / var.max() # normalize by max
        var = 1 - var # inverse
        pvar = var / np.sum(var)
        idx = np.random.choice(a=range(pvar.shape[0]), p=pvar)
        x = xx_o[idx]
        y = yy_o[idx]
        outer_pt = np.array([y, x])
        inner_pt = closest_inner_px[y, x]
        v = inner_pt - outer_pt
        magn = np.linalg.norm(v)
        unitv = v / magn
        originv = [0, 1] # [y, x]
        angle = np.arccos(np.dot(unitv, originv))
        if v[0] < 0:
            angle = -angle

        # Publish heatmap
        outer_edges_var = np.zeros((im_height, im_width))
        for i in range(xx_o.shape[0]):
            outer_edges_var[yy_o[i]][xx_o[i]] = var[i]
        outer_edges_var_u = (outer_edges_var / np.max(outer_edges_var) * 255.).astype('uint8')
        self.pub.publish(self.bridge.cv2_to_imgmsg(outer_edges_var_u))

        angle_x = closest_inner_px[:,:,1]-xx
        angle_y = -(closest_inner_px[:,:,0]-yy)
        mag = np.linalg.norm([angle_x,angle_y])
        angle_x = angle_x/mag
        angle_y = angle_y/mag

        return outer_pt[0], outer_pt[1], angle, inner_pt[0], inner_pt[1], outer_edges_var, angle_x, angle_y

class NetworkGraspSelector:
    """Grasp Selector using the output of the cloth region segmentation network
    """
    def __init__(self, grasp_point_method, grasp_angle_method):
        self.grasp_point_method = grasp_point_method
        self.grasp_angle_method = grasp_angle_method
        self.grasp_pt = None
        self.bridge = CvBridge()
        self.pub = rospy.Publisher('pheatmap', Image, queue_size=1)
    
    def winclicked(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            rospy.loginfo('Selected grasp point y: %d x: %d' % (y, x))
            self.grasp_pt = [y, x]

    def sample_grasp(self, segmentation, pred):
        """Takes a 2D array prop to prob as input and sample grasp point."""
        # Filter for outer edge points w/o overlap only
        im_height, im_width, _ = segmentation.shape
        outer_edges_mask = np.zeros((im_height, im_width))
        outer_edges_mask[segmentation[:,:,1]==255] = 1

        var_map = outer_edges_mask*pred[:, :, -1]
        pvec = var_map.ravel()/np.sum(var_map) # flatten and normalize to PMF
        idx = np.random.choice(a=range(pvec.shape[0]), p=pvec)
        y,x = np.unravel_index(idx, var_map.shape)

        im = (var_map / var_map.max() * 255).astype(np.uint8)
        self.pub.publish(self.bridge.cv2_to_imgmsg(im, encoding="mono8"))

        return np.array([y,x])

    def select_grasp(self, rgb, corners, outer_edges, inner_edges, pred, retries=1, num_neighbour=8):
        impred = np.zeros((corners.shape[0], corners.shape[1], 3), dtype=np.uint8)
        impred[:, :, 0] += corners
        impred[:, :, 1] += outer_edges
        impred[:, :, 2] += inner_edges

        idxs = np.where(corners == 255)
        corners[:] = 1
        corners[idxs] = 0
        idxs = np.where(outer_edges == 255)
        outer_edges[:] = 1
        outer_edges[idxs] = 0
        idxs = np.where(inner_edges == 255)
        inner_edges[:] = 1
        inner_edges[idxs] = 0

        # Choose pixel in pred to grasp
        grasp_target = rospy.get_param('grasp_target')
        channel = 1 if grasp_target == 'edges' else 0
        indices = np.where(impred[:, :, channel] == 255) # outer_edge
        if len(indices[0]) == 0:
            if rospy.get_param("break_on_error"):
                raise rospy.ServiceException('No graspable pixels detected')
            else: 
                rospy.logerr('No graspable pixels detected')
                return 0, 0, 0, 0, 0

        if self.grasp_point_method == 'policy':
            outer_edges = deepcopy(pred[:, :, 1])
            mask = np.zeros_like(outer_edges)
            mask[outer_edges > 0.9] = 1
            var = deepcopy(pred[:, :, -1])
            var *= mask

            pvar = var.ravel()/np.sum(var) # flatten and normalize to PMF
            idx = np.random.choice(a=range(pvar.shape[0]), p=pvar)
            y, x = np.unravel_index(idx, var.shape)

            var_map = (var / var.max() * 255.).astype('uint8')
            self.pub.publish(self.bridge.cv2_to_imgmsg(var_map))
        else: 
            if self.grasp_point_method == 'manual':
                # Only works once due to rendering issues, need to restart service
                rospy.logwarn("Manually choosing grasp point")
                wintitle = 'Choose grasp point'
                cv2.namedWindow(wintitle)
                cv2.setMouseCallback(wintitle, self.winclicked)
                cv2.imshow(wintitle, impred)
                cv2.waitKey(0)
                y, x = self.grasp_pt
            elif self.grasp_point_method == 'random':
                idx = np.random.choice(range(len(indices[0])))
                y = indices[0][idx]
                x = indices[1][idx]
            elif self.grasp_point_method == 'confidence':
                # Filter out ambiguous points
                # impred:[im_height, im_width, 3] -> corner, outer edge, inner edge predictions
                segmentation = deepcopy(impred)
                im_height, im_width, _ = segmentation.shape
                segmentation[np.logical_and(impred[:,:,1]==255, impred[:,:,2]==255),2] = 0
                segmentation[np.logical_and(impred[:,:,1]==255, impred[:,:,2]==255),1] = 0

                inner_edges_filt = np.ones((im_height, im_width))

                inner_edges_filt[segmentation[:,:,2]==255] = 0

                # Get outer-inner edge correspondence
                xx, yy =  np.meshgrid([x for x in range(im_width)],
                                    [y for y in range(im_height)])
                if grasp_target == 'edges':
                    xx_o = xx[segmentation[:,:,1]==255]
                    yy_o = yy[segmentation[:,:,1]==255]
                else:
                    xx_o = xx[segmentation[:,:,0]==255]
                    yy_o = yy[segmentation[:,:,0]==255]

                xx_i = xx[segmentation[:,:,2]==255]
                yy_i = yy[segmentation[:,:,2]==255]

                _, lbl = cv2.distanceTransformWithLabels(inner_edges_filt.astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)

                loc = np.where(inner_edges_filt==0)
                xx_inner = loc[1]
                yy_inner = loc[0]
                label_to_loc = [[0,0]]

                for j in range(len(yy_inner)):
                    label_to_loc.append([yy_inner[j],xx_inner[j]])

                label_to_loc = np.array(label_to_loc)
                direction = label_to_loc[lbl]
                # Calculate distance to the closest inner edge point for every pixel in the image
                distance = np.zeros(direction.shape)

                distance[:,:,0] = np.abs(direction[:,:,0]-yy)
                distance[:,:,1] = np.abs(direction[:,:,1]-xx)
                
                # Normalize distance vectors
                mag = np.linalg.norm([distance[:,:,0],distance[:,:,1]],axis = 0)+0.00001
                distance[:,:,0] = distance[:,:,0]/mag
                distance[:,:,1] = distance[:,:,1]/mag

                # Get distances of outer edges
                distance_o = distance[segmentation[:,:,1]==255,:]

                # Get outer edge neighbors of each outer edge point
                num_neighbour = 100

                # For every outer edge point, find its closest K neighbours 
                tree = KDTree(np.vstack([xx_o,yy_o]).T, leaf_size=2)
                dist, ind = tree.query(np.vstack([xx_o,yy_o]).T, k=num_neighbour)
                
                xx_neighbours = distance_o[ind][:,:,1]
                yy_neighbours = distance_o[ind][:,:,0]
                xx_var = np.var(xx_neighbours,axis = 1)
                yy_var = np.var(yy_neighbours,axis = 1)
                var = xx_var+yy_var
                var = (var-np.min(var))/(np.max(var)-np.min(var))
                
                # Choose min var point
                var_min = np.min(var)
                min_idxs = np.where(var == var_min)[0]
                rospy.loginfo("Number of min var indices: %d" %len(min_idxs))
                idx = np.random.choice(min_idxs)
                x = xx_o[idx]
                y = yy_o[idx]
            else:
                raise NotImplementedError

        # Get outer_pt and inner_pt for computing grasp angle
        if self.grasp_angle_method == 'inneredge':
            temp, lbl = cv2.distanceTransformWithLabels(inner_edges.astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
            loc = np.where(inner_edges==0)
            xx_inner = loc[1]
            yy_inner = loc[0]
            label_to_loc = zip(yy_inner, xx_inner)
            label_to_loc.insert(0, (0, 0)) # 1-indexed
            label_to_loc = np.array(label_to_loc)
            direction = label_to_loc[lbl]
            outer_pt = np.array([y, x])
            inner_pt = direction[y, x]
        elif self.grasp_angle_method == 'center':
            idx = np.random.choice(range(len(indices[0])))
            y = indices[0][idx]
            x = indices[1][idx]
            
            bbox = deepcopy(outer_edges) if grasp_target == 'edges' else deepcopy(corners)
            idxs = np.where(bbox == 0)
            bbox[:] = 0
            bbox[idxs] = 1
            
            pbox = cv2.boundingRect(bbox)
            center_x = pbox[0] + 0.5*pbox[2]
            center_y = pbox[1] + 0.5*pbox[3]
            outer_pt = np.array([y, x])
            inner_pt = np.array([center_y, center_x])
        else:
            raise NotImplementedError

        v = inner_pt - outer_pt
        magn = np.linalg.norm(v)

        if magn == 0:
            error_msg = "magnitude is zero for %d samples" % retries
            if rospy.get_param('break_on_error'):
                raise rospy.ServiceException(error_msg)
            else:
                rospy.logerr(error_msg)
                magn = 1.0

        unitv = v / magn
        originv = [0, 1] # [y, x]
        angle = np.arccos(np.dot(unitv, originv))

        if v[0] < 0:
            angle = -angle

        return outer_pt[0], outer_pt[1], angle, inner_pt[0], inner_pt[1]

    def plot(impred, xx_o, yy_o, var, outer_edges_filt, xx, yy, segmentation):
        """Plot for debugging
        """
        impred2 = deepcopy(segmentation)
        impred2[:, :, 0] = 0
        fig = plt.figure()
        ax = plt.subplot(121)
        empty = np.zeros(impred.shape)
        ax.imshow(empty)
        scat = ax.scatter(xx_o, yy_o, c=var, cmap='RdBu', s=3)
        plt.colorbar(scat)
        ax.scatter(x, y, c='blue', alpha=0.7)

        ax = plt.subplot(122)
        ax.imshow(impred2)

        factor = 2
        xx = xx[outer_edges_filt==0]
        yy = yy[outer_edges_filt==0]
        direction_o = direction[segmentation[:,:,1]==255,:]
        ax.quiver(xx_o[::factor],yy_o[::factor],direction_o[::factor,1]-xx_o[::factor],-direction_o[::factor,0]+yy_o[::factor], color='white', scale=1, scale_units='x')

        base_path = "/media/ExtraDrive1/clothfolding/uncertainty"
        tstamp = datetime.now().strftime("%d_%m_%Y_%H:%M:%S")
        tstamp_path = os.path.join(base_path, tstamp)
        os.makedirs(tstamp_path)

        fig.canvas.draw()
        w,h = fig.canvas.get_width_height()
        buf = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
        np.save(os.path.join(tstamp_path, "plot_%s" % tstamp), buf)
        
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(tstamp_path, "rgb_%s.png" % tstamp), rgb)
        plt.savefig(os.path.join(tstamp_path, 'uncertainty_%s.png' % tstamp))
        plt.show()

class ClothSegGraspSelector:
    """
    Grasp selector using depth-based cloth segmentation.
    """
    def select_grasp(self, prediction, thresh=150):
        # Compute gradients and gradient directions
        grad_x = cv2.Scharr(prediction.astype(np.float32), cv2.CV_32F, dx=1, dy=0)
        grad_y = cv2.Scharr(prediction.astype(np.float32), cv2.CV_32F, dx=0, dy=1)
        abs_grad_x = cv2.convertScaleAbs(grad_x, alpha=255/grad_x.max())
        abs_grad_y = cv2.convertScaleAbs(grad_y, alpha=255/grad_y.max())
        grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
        
        # Choose random edge_pixel
        edge_pixels = np.where(grad > thresh)
        idx = np.random.choice(range(len(edge_pixels[0])))
        py = edge_pixels[0][idx]
        px = edge_pixels[1][idx]
        _, angles = cv2.cartToPolar(grad_x, grad_y)
        angle = angles[py, px]
        return py, px, angle

class CannyGraspSelector():
    """
    Grasp selector using depth-based canny edge detection.
    """
    def select_grasp(self, detection, thresh=200):
        edges = detection[:, :, 0]
        grad_x = detection[:, :, 1]
        grad_y = detection[:, :, 2]
        edge_pixels = np.where(edges > thresh)
        idx = np.random.choice(range(len(edge_pixels[0])))
        py = edge_pixels[0][idx]
        px = edge_pixels[1][idx]
        _, angles = cv2.cartToPolar(grad_x, grad_y)
        angle = angles[py][px] - np.pi
        return py, px, angle

class CannyColorGraspSelector():
    """
    Grasp selector using color-based canny edge detection.
    """
    def select_grasp(self, detection, thresh=200):
        edges = detection[:, :, 0]
        grad_x = detection[:, :, 1]
        grad_y = detection[:, :, 2]
        edge_pixels = np.where(edges > thresh)
        idx = np.random.choice(range(len(edge_pixels[0])))
        py = edge_pixels[0][idx]
        px = edge_pixels[1][idx]
        _, angles = cv2.cartToPolar(grad_x, grad_y)
        angle = angles[py][px]
        return py, px, angle

class HarrisColorGraspSelector():
    """
    Grasp selector using color-based harris corner detection.
    """
    def select_grasp(self, detection):
        corner_pred = detection[:, :, 0]
        angles = detection[:, :, 1]
        py, px = np.unravel_index(np.argmax(corner_pred, axis=None), corner_pred.shape)
        angle = angles[py][px]
        return py, px, angle

class HarrisGraspSelector():
    """
    Grasp selector using depth-based harris corner detection.
    """
    def select_grasp(self, detection):
        corner_pred = detection[:, :, 0]
        angles = detection[:, :, 1]
        py, px = np.unravel_index(np.argmax(corner_pred, axis=None), corner_pred.shape)
        angle = angles[py][px] - np.pi
        return py, px, angle

class GraspSelector():
    """
    Runs service that selects grasps.
    """
    def __init__(self):
        rospy.init_node('selectgrasp_service')
        self.bridge = CvBridge()

        self.detection_method = rospy.get_param('detection_method')
        self._init_selector()

        self.server = rospy.Service('select_grasp', SelectGrasp, self._server_cb)

    def _init_selector(self):
        if self.detection_method == 'groundtruth':
            self.selector = GroundTruthSelector()
        elif self.detection_method == 'network':
            grasp_point_method = rospy.get_param('grasp_point_method')
            grasp_angle_method = rospy.get_param('grasp_angle_method')
            self.selector = NetworkGraspSelector(grasp_point_method, grasp_angle_method)
        elif self.detection_method == 'clothseg':
            self.selector = ClothSegGraspSelector()
        elif self.detection_method == 'canny':
            self.selector = CannyGraspSelector()
        elif self.detection_method == 'canny_color':
            self.selector = CannyColorGraspSelector()
        elif self.detection_method == 'harris':
            self.selector = HarrisGraspSelector()
        elif self.detection_method == 'harris_color':
            self.selector = HarrisColorGraspSelector()
        else:
            raise NotImplementedError

    def _server_cb(self, req):
        rospy.loginfo('Received grasp selection request')
        inner_py = None
        inner_px = None

        if self.detection_method == 'groundtruth':
            pred = deepcopy(self.bridge.imgmsg_to_cv2(req.prediction))
            py, px, angle, inner_py, inner_px, var, angle_x, angle_y = self.selector.select_grasp(pred)
        elif self.detection_method == 'network':
            corners = deepcopy(self.bridge.imgmsg_to_cv2(req.corners))
            outer_edges = deepcopy(self.bridge.imgmsg_to_cv2(req.outer_edges))
            inner_edges = deepcopy(self.bridge.imgmsg_to_cv2(req.inner_edges))
            rgb = deepcopy(self.bridge.imgmsg_to_cv2(req.rgb))
            pred = deepcopy(self.bridge.imgmsg_to_cv2(req.prediction))
            py, px, angle, inner_py, inner_px = self.selector.select_grasp(rgb, corners, outer_edges, inner_edges, pred)
        else:
            prediction = deepcopy(self.bridge.imgmsg_to_cv2(req.prediction))
            py, px, angle = self.selector.select_grasp(prediction)
            print(py, px, angle)

        rospy.loginfo('Sending grasp selection response py: %f, px: %f, angle: %f' % (py, px, angle))
        response = SelectGraspResponse()
        response.py = py
        response.px = px
        response.angle = angle
        if inner_py != None and inner_px != None:
            rospy.loginfo('inner_py: %f, inner_px: %f' % (inner_py, inner_px))
            response.inner_py = inner_py
            response.inner_px = inner_px
        if self.detection_method == 'groundtruth' and px != 0:
            response.var = var.flatten()
            response.angle_x = angle_x.flatten()
            response.angle_y = angle_y.flatten()

        return response

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    g = GraspSelector()
    g.run()