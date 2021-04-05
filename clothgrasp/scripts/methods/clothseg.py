#!/usr/env/bin python
# import matplotlib.pyplot as plt
import cv2
import numpy as np
import open3d as o3d
from pyquaternion import Quaternion as Quat
from copy import deepcopy

class ClothSegmenter:
    def __init__(self, D, K, w2c_pose, segment_table, table_plane, crop_dims):
        self.D = D
        self.K = K
        self.w2c_pose = w2c_pose
        self.segment_table = segment_table
        self.table_plane = table_plane
        self.crop_dims = crop_dims

    def predict(self, depth, kernel_size=5, plane_tol=0.004):
        row_start, row_end, col_start, col_end, step = self.crop_dims
        depth = depth[row_start:row_end:step, col_start:col_end:step]

        # Fill holes
        zeros = np.where(depth == 0)
        mask = np.zeros_like(depth, np.uint8)
        mask[zeros] = 1
        depth = cv2.inpaint(depth, mask, 3, cv2.INPAINT_NS)

        fx = self.K[0, 0]
        fy = self.K[1, 1]
        cx = self.K[0, 2]
        cy = self.K[1, 2]

        # Get 3D point from 2D pixels
        y_size, x_size = depth.shape[:2]
        xmap, ymap = np.meshgrid(np.arange(x_size), np.arange(y_size))
        u = xmap.reshape(-1,1).astype(np.float32)
        v = ymap.reshape(-1,1).astype(np.float32)
        z = depth.reshape(-1,1).astype(np.float32)

        # Undistort 3D points
        points = np.hstack([u + col_start, v + row_start])
        depth_undistort = cv2.undistortPoints(np.expand_dims(points, axis=0), self.K, self.D, P=self.K)
        depth_undistort = np.squeeze(depth_undistort)
        u_undistort, v_undistort = zip(*depth_undistort)
        u_undistort = np.array(u_undistort)
        v_undistort = np.array(v_undistort)
        x_undistort = (np.expand_dims(u_undistort, axis=1)-cx)*z/fx
        y_undistort = (np.expand_dims(v_undistort, axis=1)-cy)*z/fy

        # Make point cloud
        cloud = np.concatenate((x_undistort, y_undistort, z), axis=1).astype(np.float64)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cloud)

        # Segment plane using RANSAC
        if self.segment_table:
            plane, inliers = pcd.segment_plane(plane_tol, 10, 200)
            [a, b, c, d] = plane
            print("Plane model [a, b, c, d]: %f %f %f %f" % (a, b, c, d))
            return None
        else:
            points = np.array(pcd.points).astype(np.float64)
            inliers = []
            for idx in range(points.shape[0]):
                point = np.append(points[idx], 1)
                distance = np.abs(self.table_plane.dot(point))
                if distance < plane_tol:
                    inliers.append(idx)

        pcd_outliers = pcd.select_down_sample(inliers, invert=True)
        points = np.asarray(pcd_outliers.points).astype(np.float64)

        # Reconstruct depth image
        depth_img = np.zeros(depth.shape)
        if len(points) != 0:
            imagePoints, _ = cv2.projectPoints(points, (0, 0, 0), (0, 0, 0), self.K, self.D)
            imagePoints = imagePoints[:, 0, :]
            imagePoints = np.append(imagePoints, np.zeros((imagePoints.shape[0], 1)), axis=1)
            depths = points[:,2]
            for i in range(imagePoints.shape[0]):
                d = depths[i]
                py = int(np.rint(imagePoints[i, 1])) - row_start
                px = int(np.rint(imagePoints[i, 0])) - col_start
                if py < depth.shape[0] and px < depth.shape[1]:
                    depth_img[py, px] = d

        # Create segmask
        mask_img = np.zeros(depth_img.shape)
        mask_img[depth_img > 0] = 1

        kernel = np.ones((kernel_size, kernel_size),np.uint8)
        mask_img = cv2.morphologyEx(mask_img, cv2.MORPH_CLOSE, kernel)
        mask_img = cv2.morphologyEx(mask_img, cv2.MORPH_OPEN, kernel)

        # plt.imshow(mask_img, cmap='gray')
        # plt.show()

        return mask_img

