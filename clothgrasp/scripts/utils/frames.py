import os
import numpy as np 
import rospy
import matplotlib.pyplot as plt
import open3d as o3d
import cv2
from PIL import Image
from geometry_msgs.msg import Pose, Point32
import tf
from pyquaternion import Quaternion as Quat
from visualization_msgs.msg import Marker
from sensor_msgs.msg import PointCloud

rospy.init_node('frames')

base_path = '/media/ExtraDrive2/clothfolding/data_studio_painted_towel_r0.1/camera_3'
depth_path = os.path.join(base_path, '0_depth.npy')
rgb_path = os.path.join(base_path, 'rgb_0.png')

depth = np.load(depth_path)
rgb = Image.open(rgb_path)

D = [0.6862431764602661, -2.8917713165283203, 0.0004992862232029438, -4.462565993890166e-05, 1.6113708019256592, 0.5638872385025024, -2.7169768810272217, 1.540696382522583]
D = np.array(D)
K = [611.1693115234375, 0.0, 638.8131713867188, 0.0, 611.1657104492188, 367.74871826171875, 0.0, 0.0, 1.0]
K = np.array(K).reshape((3, 3))

# Fill holes
zeros = np.where(depth == 0)
mask = np.zeros_like(depth, np.uint8)
mask[zeros] = 1
depth = cv2.inpaint(depth, mask, 3, cv2.INPAINT_NS)

fx = K[0, 0]
fy = K[1, 1]
cx = K[0, 2]
cy = K[1, 2]

# Get 3D point from 2D pixels
y_size, x_size = depth.shape[:2]
print(y_size, x_size)
xmap, ymap = np.meshgrid(np.arange(x_size), np.arange(y_size))
u = xmap.reshape(-1,1).astype(np.float32)
v = ymap.reshape(-1,1).astype(np.float32)
z = depth.reshape(-1,1).astype(np.float32)

# Undistort 3D points
col_start, row_start = [450, 180]
points = np.hstack([u + col_start, v + row_start])
depth_undistort = cv2.undistortPoints(np.expand_dims(points, axis=0), K, D, P=K)
depth_undistort = np.squeeze(depth_undistort)
u_undistort, v_undistort = zip(*depth_undistort)
u_undistort = np.array(u_undistort)
v_undistort = np.array(v_undistort)
x = (np.expand_dims(u_undistort, axis=1)-cx)*z/fx
y = (np.expand_dims(v_undistort, axis=1)-cy)*z/fy

# Make point cloud
cloud = np.concatenate((x, y, z), axis=1).astype(np.float64)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(cloud)
# o3d.visualization.draw_geometries([pcd])

midy = int(y_size / 2)
midx = int(x_size / 2)
points = np.array([[midx + col_start, midy + row_start]], dtype=np.float32)
pt_undistorted = cv2.undistortPoints(np.expand_dims(points, axis=0), K, D, P=K)
pt_undistorted = np.squeeze(pt_undistorted, axis=0)
pz = depth[midy][midx]
px = (pt_undistorted[0, 0] - cx) / fx * pz
py = (pt_undistorted[0, 1] - cy) / fy * pz

print(px, py, pz)
midpose = Pose()
midpose.position.x = px
midpose.position.y = py
midpose.position.z = pz

def get_marker(pose, frame_id, color=(1.0, 0.0, 0.0)):
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = rospy.Time.now()
    marker.id = 0
    marker.type = 2 # sphere
    marker.action = 0 # add/modify
    marker.pose = pose
    marker.color.r = color[0]
    marker.color.g = color[1]
    marker.color.b = color[2]
    marker.color.a = 1.0
    marker.scale.x = 0.05
    marker.scale.y = 0.05
    marker.scale.z = 0.05
    marker.lifetime = rospy.Duration(0)
    return marker

marker = get_marker(midpose, 'camera')

# transform pose
w2c_pose = [0.6679, 0.0302, 0.6407, 0.70277, 0.70456, -0.07513, -0.06369]

qcam = Quat(x=w2c_pose[3], y=w2c_pose[4], z=w2c_pose[5], w=w2c_pose[6]).unit
T_world2cam = qcam.transformation_matrix
T_world2cam[0:3, 3] = w2c_pose[0:3]

midpt = T_world2cam.dot([px, py, pz, 1])
midpose_w = Pose()
midpose_w.position.x = midpt[0]
midpose_w.position.y = midpt[1]
midpose_w.position.z = midpt[2]
marker = get_marker(midpose_w, 'world', color=(0.0, 1.0, 0.0))

quat = tf.transformations.quaternion_from_euler(0, np.pi, -np.pi/2)

pc = PointCloud()
pc.header.frame_id = 'camera'
pc.header.stamp = rospy.Time.now()
for x, y, z in cloud:
    point = Point32()
    point.x = x
    point.y = y
    point.z = z
    pc.points.append(point)

imagePoints, _ = cv2.projectPoints(cloud, (0, 0, 0), (0, 0, 0), K, D) 
print(imagePoints[0:3], imagePoints.shape)
im = np.zeros((480, 640))
imagePoints = np.rint(imagePoints).squeeze().astype(int)
for px, py in imagePoints:
    im[py][px] = 1.0
plt.imshow(im)
plt.show()

markerpub = rospy.Publisher('marker', Marker, queue_size=1)
cloudpub = rospy.Publisher('cloud', PointCloud, queue_size=1)
br = tf.TransformBroadcaster()
while not rospy.is_shutdown():
    br.sendTransform(w2c_pose[:3],
                     w2c_pose[3:],
                     rospy.Time.now(),
                     "camera",
                     "world")
    
    br.sendTransform([midpt[0], midpt[1], 1.0],
                     quat,
                     rospy.Time.now(),
                     "topdown",
                     "world")

    markerpub.publish(marker)
    cloudpub.publish(pc)
    
