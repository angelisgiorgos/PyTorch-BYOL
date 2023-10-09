from typing import Any
import cv2
import numpy as np

class SMPLPoseFlipping(object):
    """Flip SMPL pose parameters horizontally.

    Args:
        pose (np.ndarray([72])): SMPL pose parameters
    Returns:
        pose_flipped
    """
    def __init__(self,):
        pass

    def __call__(self, pose):
        flippedParts = [
        0, 1, 2, 6, 7, 8, 3, 4, 5, 9, 10, 11, 15, 16, 17, 12, 13, 14, 18, 19,
        20, 24, 25, 26, 21, 22, 23, 27, 28, 29, 33, 34, 35, 30, 31, 32, 36, 37,
        38, 42, 43, 44, 39, 40, 41, 45, 46, 47, 51, 52, 53, 48, 49, 50, 57, 58,
        59, 54, 55, 56, 63, 64, 65, 60, 61, 62, 69, 70, 71, 66, 67, 68]
        pose_flipped = pose[flippedParts]
        # Negate the second and the third dimension of the axis-angle
        pose_flipped[1::3] = -pose_flipped[1::3]
        pose_flipped[2::3] = -pose_flipped[2::3]
        return pose_flipped
    

class SMPLPoseRotating(object):
    """Rotate SMPL pose parameters.

    SMPL (https://smpl.is.tue.mpg.de/) is a 3D
    human model.
    Args:
        pose (np.ndarray([72])): SMPL pose parameters
        rot (float): Rotation angle (degree).
    Returns:
        pose_rotated
    """
    def __init__(self, rot):
        self.rot = rot

    def __call__(self, pose):
        pose_rotated = pose.copy()
        if self.rot != 0:
            rot_mat = _construct_rotation_matrix(-self.rot)
            orient = pose[:3]
            # find the rotation of the body in camera frame
            per_rdg, _ = cv2.Rodrigues(orient.astype(np.float32))
            # apply the global rotation to the global orientation
            res_rot, _ = cv2.Rodrigues(np.dot(rot_mat, per_rdg))
            pose_rotated[:3] = (res_rot.T)[0]
        pose_rotated = np.expand_dims(pose_rotated, axis=1)
        return pose_rotated
    



def _construct_rotation_matrix(rot, size=3):
    """Construct the in-plane rotation matrix.

    Args:
        rot (float): Rotation angle (degree).
        size (int): The size of the rotation matrix.
            Candidate Values: 2, 3. Defaults to 3.
    Returns:
        rot_mat (np.ndarray([size, size]): Rotation matrix.
    """
    rot_mat = np.eye(size, dtype=np.float32)
    if rot != 0:
        rot_rad = np.deg2rad(rot)
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0, :2] = [cs, -sn]
        rot_mat[1, :2] = [sn, cs]

    return rot_mat