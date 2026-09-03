import numpy as np
from scipy.spatial.transform import Rotation as R

def Rx(angle):
    """3x3 Rotation matrix around X-axis."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [1, 0,  0],
        [0, c, -s],
        [0, s,  c]
    ])

def Ry(angle):
    """3x3 Rotation matrix around Y-axis."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [ c, 0, s],
        [ 0, 1, 0],
        [-s, 0, c]
    ])

def Rz(angle):
    """3x3 Rotation matrix around Z-axis."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ])

ELEMENTARY_ROTATIONS = {
    "x": Rx,
    "y": Ry,
    "z": Rz,
}

def set_body_orientation(data, R_matrix):
    """Converts a 3x3 rotation matrix to quaternion [w, x, y, z] for MuJoCo."""
    quat = R.from_matrix(R_matrix).as_quat()  # [x, y, z, w]
    mujoco_quat = np.array([quat[3], quat[0], quat[1], quat[2]])  # [w, x, y, z]
    data.qpos[3:7] = mujoco_quat
