import os
import torch
import numpy as np

import torch.nn.functional as F
import torchvision
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils import _create_model_training_folder
from utils3d import (
    TexturedIUVRenderer, SMPL, get_intrinsics_matrix, perspective_project_torch,
    smpl_based_val_representation
    )
import config

FOCAL_LENGTH = 5000
SMPL_MODEL_DIR = './additional/smpl'


def initialize_camera(batch_size, input_size, device):
    mean_cam_t = np.array([0., 0.2, 42.])
    mean_cam_t = torch.from_numpy(mean_cam_t).float().to(device)
    mean_cam_t = mean_cam_t[None, :].expand(batch_size, -1)
    cam_K = get_intrinsics_matrix(input_size, input_size, FOCAL_LENGTH)
    cam_K = torch.from_numpy(cam_K.astype(np.float32)).to(device)
    cam_K = cam_K[None, :, :].expand(batch_size, -1, -1)
    cam_R = torch.eye(3).to(device)
    cam_R = cam_R[None, :, :].expand(batch_size, -1, -1)
    return mean_cam_t, cam_K, cam_R


class BYOLTrainer:
    def __init__(self, online_network, target_network, predictor, optimizer, device, **params):
        self.online_network = online_network
        self.target_network = target_network
        self.optimizer = optimizer
        self.device = device
        self.predictor = predictor
        self.max_epochs = params['max_epochs']
        self.writer = SummaryWriter()
        self.m = params['m']
        self.batch_size = params['batch_size']
        self.num_workers = params['num_workers']
        self.checkpoint_interval = params['checkpoint_interval']
        _create_model_training_folder(self.writer, files_to_same=["./config/config.yaml", "main.py",
                                                                  'trainer.py'])
        self.mean_cam_t, self.cam_K, self.cam_R = initialize_camera(params['batch_size'],
                                                                    params['input_size'], device)
        self.pytorch3d_renderer = TexturedIUVRenderer(device=device,
                                                      batch_size=params['batch_size'],
                                                      img_wh=params['input_size'],
                                                      projection_type='perspective',
                                                      perspective_focal_length=FOCAL_LENGTH,
                                                      render_rgb=False,
                                                      bin_size=32)
        self.smpl = SMPL(SMPL_MODEL_DIR, batch_size=params['batch_size'])

    @torch.no_grad()
    def _update_target_network_parameters(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.online_network.parameters(),
                                    self.target_network.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @staticmethod
    def regression_loss(x, y):
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        return 2 - 2 * (x * y).sum(dim=-1)

    def initializes_target_network(self):
        # init momentum network as encoder net
        for param_q, param_k in zip(self.online_network.parameters(),
                                    self.target_network.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

    def process_batch_view(self, samples, second_view=False):
        target_pose = samples['pose']
        target_shape = samples['shape']
        target_pose = target_pose.to(self.device)
        target_shape = target_shape.to(self.device)
        target_smpl_output = self.smpl(body_pose=target_pose[:, 3:],
                                       global_orient=target_pose[:, :3],
                                       betas=target_shape)
        target_vertices = target_smpl_output.vertices
        target_joints_all = target_smpl_output.joints
        target_joints_coco = target_joints_all[:, config.ALL_JOINTS_TO_COCO_MAP, :]
        target_joints2d_coco = perspective_project_torch(target_joints_coco, self.cam_R,
                                                         self.mean_cam_t,
                                                         cam_K=self.cam_K)

        input_representation = smpl_based_val_representation(
            config=config,
            target_vertices=target_vertices,
            target_joints2d_coco=target_joints2d_coco,
            renderer=self.pytorch3d_renderer,
            cam_t=self.mean_cam_t,
            device=self.device,
            second_view=second_view)
        return input_representation

    def train(self, train_dataset):

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                  num_workers=self.num_workers, drop_last=False, shuffle=True)

        niter = 0
        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')

        self.initializes_target_network()

        for epoch_counter in range(self.max_epochs):

            for (batch_view_1, batch_view_2), _ in train_loader:

                batch_view_1 = self.process_batch_view(batch_view_1)
                batch_view_2 = self.process_batch_view(batch_view_2, second_view=True)

                if niter == 0:
                    grid = torchvision.utils.make_grid(batch_view_1[:32])
                    self.writer.add_image('views_1', grid, global_step=niter)

                    grid = torchvision.utils.make_grid(batch_view_2[:32])
                    self.writer.add_image('views_2', grid, global_step=niter)

                loss = self.update(batch_view_1, batch_view_2)
                self.writer.add_scalar('loss', loss, global_step=niter)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self._update_target_network_parameters()  # update the key encoder
                niter += 1

            print("End of epoch {}".format(epoch_counter))

        # save checkpoints
        self.save_model(os.path.join(model_checkpoints_folder, 'model.pth'))

    def update(self, batch_view_1, batch_view_2):
        # compute query feature
        predictions_from_view_1 = self.predictor(self.online_network(batch_view_1))
        predictions_from_view_2 = self.predictor(self.online_network(batch_view_2))

        # compute key features
        with torch.no_grad():
            targets_to_view_2 = self.target_network(batch_view_1)
            targets_to_view_1 = self.target_network(batch_view_2)

        loss = self.regression_loss(predictions_from_view_1, targets_to_view_1)
        loss += self.regression_loss(predictions_from_view_2, targets_to_view_2)
        return loss.mean()

    def save_model(self, PATH):

        torch.save({
            'online_network_state_dict': self.online_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            }, PATH)
