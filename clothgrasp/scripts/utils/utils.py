from pyquaternion import Quaternion as Quat
import numpy as np

def unit_vector_from_pose(pose):
    """Returns unit pose vector.
    """
    q = Quat(w=pose.orientation.w, x=pose.orientation.x, y=pose.orientation.y, z=pose.orientation.z).unit
    unit_v = np.array([0., 0., 1.]) # rotate world frame z axis
    pose_v = q.rotate(unit_v)
    return pose_v