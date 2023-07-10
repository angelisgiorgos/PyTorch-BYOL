import os
import config
import cv2
import torch
import torchvision
import torch.nn as nn
import numpy as np


_HumanData_SUPPORTED_KEYS = {
    'image_path': {
        'type': list,
    },
    'image_id': {
        'type': list,
    },
    'bbox_xywh': {
        'type': np.ndarray,
        'shape': (-1, 5),
        'dim': 0
    },
    'config': {
        'type': str,
        'dim': None
    },
    'keypoints2d': {
        'type': np.ndarray,
        'shape': (-1, -1, 3),
        'dim': 0
    },
    'keypoints3d': {
        'type': np.ndarray,
        'shape': (-1, -1, 4),
        'dim': 0
    },
    'smpl': {
        'type': dict,
        'slice_key': 'betas',
        'dim': 0
    },
    'smplh': {
        'type': dict,
        'slice_key': 'betas',
        'dim': 0
    },
    'smplx': {
        'type': dict,
        'slice_key': 'betas',
        'dim': 0
    },
    'meta': {
        'type': dict,
    },
    'keypoints2d_mask': {
        'type': np.ndarray,
        'shape': (-1, ),
        'dim': None
    },
    'keypoints2d_convention': {
        'type': str,
        'dim': None
    },
    'keypoints3d_mask': {
        'type': np.ndarray,
        'shape': (-1, ),
        'dim': None
    },
    'keypoints3d_convention': {
        'type': str,
        'dim': None
    },
    'vertices': {
        'type': np.ndarray,
        'shape': (-1, ),
        'dim': None
    },
    'misc': {
        'type': dict,
    },
}


class HumanImageDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        super(HumanImageDataset, self).__init__()
        self.data_path = data_path
        self.load_annotations()
        
        
    def get_annotation_file(self):
        """Get path of the annotation file."""
        self.ann_file = self.data_path
        
    def load_file(self, npz_path):
        """Load data from npz_path and update them to self.
        Args:
            npz_path (str):
                Path to a dumped npz file.
        """
        supported_keys = _HumanData_SUPPORTED_KEYS
        start_dict = {}
        with np.load(npz_path, allow_pickle=True) as npz_file:
            tmp_data_dict = dict(npz_file)
            for key, value in list(tmp_data_dict.items()):
                if isinstance(value, np.ndarray) and\
                        len(value.shape) == 0:
                    # value is not an ndarray before dump
                    value = value.item()
                elif key in supported_keys and\
                        type(value) != supported_keys[key]['type']:
                    value = supported_keys[key]['type'](value)
                if value is None:
                    tmp_data_dict.pop(key)
                elif key == '__key_strict__' or \
                        key == '__data_len__' or\
                        key == '__keypoints_compressed__':
                    self.__setattr__(key, value)
                    # pop the attributes to keep dict clean
                    tmp_data_dict.pop(key)
                elif key == 'bbox_xywh' and value.shape[1] == 4:
                    value = np.hstack([value, np.ones([value.shape[0], 1])])
                    tmp_data_dict[key] = value
                else:
                    tmp_data_dict[key] = value
            start_dict.update(tmp_data_dict)
        self.imgname = start_dict['image_path']
        return start_dict
        
    def load_annotations(self):
        """Load annotation from the annotation file."""
        self.get_annotation_file()
        self.human_data = self.load_file(self.ann_file)
        
    def pose_preprocess(self, pose, global_orient):
        pose = np.reshape(pose, (-1, ))
        pose = np.concatenate((pose, global_orient), axis=None)
        pose = pose.astype('float32')
        return pose
    
        
    def prepare_raw_data(self, idx: int):
        info = {}
            
        if 'smpl' in self.human_data:
            smpl_dict = self.human_data['smpl']
        else:
            smpl_dict = {}
            
        if 'body_pose' in smpl_dict:
            body_pose = smpl_dict['body_pose'][idx]
        else:
            body_pose = np.zeros((23, 3))

        if 'global_orient' in smpl_dict:
            global_orient = smpl_dict['global_orient'][idx]
        else:
            global_orient = np.zeros((3))

        if 'betas' in smpl_dict:
            betas = smpl_dict['betas'][idx]
        else:
            betas = np.zeros((10))

        if 'transl' in smpl_dict:
            smpl_transl = smpl_dict['transl'][idx]
        else:
            smpl_transl = np.zeros((3))
        pose = self.pose_preprocess(body_pose, global_orient)
        info['pose'] = torch.from_numpy(pose).float()
        info['shape'] = torch.from_numpy(betas).float()
        return info
    
    def __getitem__(self, index):
        return self.prepare_raw_data(index)
    
    
    def __len__(self):
        return len(self.imgname)

