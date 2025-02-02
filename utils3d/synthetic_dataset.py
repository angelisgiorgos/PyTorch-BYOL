import numpy as np
import torch
import cv2
import os
from PIL import Image
from matplotlib import cm

class SyntheticTrainingDataset(torch.utils.data.Dataset):
    """
    Simple dataset that loads a npz file with 3 arrays containing:
    1) file names (from original datasets)
    2) SMPL pose parameters
    3) SMPL shape parameters.

    Returns dict with SMPL pose and shape (not augmented) as torch tensors.
    """
    def __init__(self,
<<<<<<< HEAD
                 npz_path,
                 textures_path,
                 backgrounds_dir_path,
                 transforms=None,
                 background_transforms =None,
                 params_from='all',
                 grey_tex_prob=0.05,
                 img_wh=256):
        
=======
                 npz_path='/data/angelisg/aicc-Avatar-Body-Capture/datasets/amass_up3d_3dpw_train.npz',
                 transforms=None,
                 params_from='all'):
>>>>>>> 3c27d33b631e7063f55844bab34aa02b94b5fea6

        assert params_from in ['all', 'h36m', 'up3d', '3dpw', 'not_amass']

        self.transforms = transforms
        self.background_transforms = background_transforms

        data = np.load(npz_path)
        self.fnames = data['fnames']
        self.poses = data['poses']
        self.shapes = data['shapes']
<<<<<<< HEAD
        self.img_wh = img_wh
=======
        self.transforms = transforms
>>>>>>> 3c27d33b631e7063f55844bab34aa02b94b5fea6

        if params_from != 'all':
            if params_from == 'not_amass':
                indices = [i for i, x in enumerate(self.fnames)
                           if x.startswith('h36m') or x.startswith('up3d')
                           or x.startswith('3dpw')]
                self.fnames = [self.fnames[i] for i in indices]
                self.poses = [self.poses[i] for i in indices]
                self.shapes = [self.shapes[i] for i in indices]
            else:
                indices = [i for i, x in enumerate(self.fnames) if x.startswith(params_from)]
                self.fnames = [self.fnames[i] for i in indices]
                self.poses = [self.poses[i] for i in indices]
                self.shapes = [self.shapes[i] for i in indices]
        
        textures = np.load(textures_path)
        self.grey_textures = textures['grey']
        self.nongrey_textures = textures['nongrey']
        self.grey_tex_prob = grey_tex_prob
        # Load LSUN backgrounds
        self.backgrounds_paths = sorted([os.path.join(backgrounds_dir_path, f)
                                         for f in os.listdir(backgrounds_dir_path)
                                         if f.endswith('.jpg')])

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        if isinstance(index, list):
            num_samples = len(index)
        else:
            num_samples = 1

        pose = self.poses[index]
        shape = self.shapes[index]
        assert pose.shape == (72,) and shape.shape == (10,), \
            "Poses and shapes are wrong: {}, {}, {}".format(self.fnames[index],
                                                            pose.shape, shape.shape)

        pose = pose.astype(np.float32)
        if self.transforms is not None:
<<<<<<< HEAD
            new_pose = self.transforms(pose)
            new_pose = np.array(new_pose)
            pose = new_pose[0, :, :]
            new_pose = new_pose[1, :, :]
            new_pose = torch.from_numpy(new_pose).squeeze(1)
            pose = torch.from_numpy(pose).squeeze(1)
=======
            pose = self.transforms(pose)
>>>>>>> 3c27d33b631e7063f55844bab34aa02b94b5fea6
        else:
            pose = torch.from_numpy(pose)
        shape = torch.from_numpy(shape.astype(np.float32))
        sample = {'pose': pose,
                'shape': shape}

        # Randomly sample texture
        texture_samples = []
        for _ in range(num_samples):
            if torch.rand(1).item() < self.grey_tex_prob:
                tex_idx = torch.randint(low=0, high=len(self.grey_textures), size=(1,)).item()
                texture = self.grey_textures[tex_idx]
            else:
                tex_idx = torch.randint(low=0, high=len(self.nongrey_textures), size=(1,)).item()
                texture = self.nongrey_textures[tex_idx]
            texture_samples.append(texture)
        texture_samples = np.stack(texture_samples, axis=0).squeeze()
        assert texture_samples.shape[-3:] == (1200, 800, 3), "Texture shape is wrong: {}".format(texture_samples.shape)
        sample['texture'] = torch.from_numpy(texture_samples / 255.).float()  # (1200, 800, 3) or (num samples, 1200, 800, 3)

        # Randomly sample background if rendering RGB
        bg_samples = []
        for _ in range(num_samples):
            bg_idx = torch.randint(low=0, high=len(self.backgrounds_paths), size=(1,)).item()
            bg_path = self.backgrounds_paths[bg_idx]
            background = cv2.cvtColor(cv2.imread(bg_path), cv2.COLOR_BGR2RGB)
            background = cv2.resize(background, (self.img_wh, self.img_wh), interpolation=cv2.INTER_LINEAR)
            background = background.transpose(2, 0, 1)
            bg_samples.append(background)
        bg_samples = np.stack(bg_samples, axis=0).squeeze()
        assert bg_samples.shape[-3:] == (3, self.img_wh, self.img_wh), "BG shape is wrong: {}".format(sample['background'].shape)
        sample['background'] = torch.from_numpy(bg_samples / 255.).float()  # (3, img_wh, img_wh) or (num samples, 3, img_wh, img_wh)

        im = Image.fromarray(np.uint8(bg_samples.transpose(1, 2, 0))).convert('RGB')


        new_bg_samples = self.background_transforms(im)
        sample['background'] = new_bg_samples[0]
        second_sample = {'pose': new_pose, 
                         'shape': shape,
                         'texture':texture,
                         'background':new_bg_samples[1]}

        return sample, second_sample