import numpy as np
import torch


def undo_keypoint_normalisation(normalised_keypoints, img_wh):
    """
    Converts normalised keypoints from [-1, 1] space to pixel space i.e. [0, img_wh]
    """
    keypoints = (normalised_keypoints + 1) * (img_wh/2.0)
    return keypoints


def check_joints2d_visibility(joints2d, img_wh):
    vis = np.ones(joints2d.shape[1])
    vis[joints2d[0] > img_wh] = 0
    vis[joints2d[1] > img_wh] = 0
    vis[joints2d[0] < 0] = 0
    vis[joints2d[1] < 0] = 0

    return vis


def check_joints2d_visibility_torch(joints2d, img_wh):
    """
    Checks if 2D joints are within the image dimensions.
    """
    vis = torch.ones(joints2d.shape[:2], device=joints2d.device, dtype=torch.bool)
    vis[joints2d[:, :, 0] > img_wh] = 0
    vis[joints2d[:, :, 1] > img_wh] = 0
    vis[joints2d[:, :, 0] < 0] = 0
    vis[joints2d[:, :, 1] < 0] = 0

    return vis



def get_spin_joint_names():
    return [
        'OP Nose',        # 0
        'OP Neck',        # 1
        'OP RShoulder',   # 2
        'OP RElbow',      # 3
        'OP RWrist',      # 4
        'OP LShoulder',   # 5
        'OP LElbow',      # 6
        'OP LWrist',      # 7
        'OP MidHip',      # 8
        'OP RHip',        # 9
        'OP RKnee',       # 10
        'OP RAnkle',      # 11
        'OP LHip',        # 12
        'OP LKnee',       # 13
        'OP LAnkle',      # 14
        'OP REye',        # 15
        'OP LEye',        # 16
        'OP REar',        # 17
        'OP LEar',        # 18
        'OP LBigToe',     # 19
        'OP LSmallToe',   # 20
        'OP LHeel',       # 21
        'OP RBigToe',     # 22
        'OP RSmallToe',   # 23
        'OP RHeel',       # 24
        'rankle',         # 25
        'rknee',          # 26
        'rhip',           # 27
        'lhip',           # 28
        'lknee',          # 29
        'lankle',         # 30
        'rwrist',         # 31
        'relbow',         # 32
        'rshoulder',      # 33
        'lshoulder',      # 34
        'lelbow',         # 35
        'lwrist',         # 36
        'neck',           # 37
        'headtop',        # 38
        'hip',            # 39 'Pelvis (MPII)', # 39
        'thorax',         # 40 'Thorax (MPII)', # 40
        'Spine (H36M)',   # 41
        'Jaw (H36M)',     # 42
        'Head (H36M)',    # 43
        'nose',           # 44
        'leye',           # 45 'Left Eye', # 45
        'reye',           # 46 'Right Eye', # 46
        'lear',           # 47 'Left Ear', # 47
        'rear',           # 48 'Right Ear', # 48
    ]

