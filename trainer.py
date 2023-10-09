import os
import torch
import numpy as np
<<<<<<< HEAD
=======

>>>>>>> 3c27d33b631e7063f55844bab34aa02b94b5fea6
import torch.nn.functional as F
import torchvision
import config
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils3d.canny_edge_detector import CannyEdgeDetector

from utils import _create_model_training_folder
from utils3d import (SMPL,
                     perspective_project_torch,
                     get_intrinsics_matrix, 
                     TexturedIUVRenderer, 
                     convert_densepose_to_6part_lsp_labels, 
                     batch_crop_seg_to_bounding_box, 
                     batch_resize, 
                     convert_2Djoints_to_gaussian_heatmaps_torch,
                     batch_add_rgb_background,
                     check_joints2d_visibility_torch,
                     random_extreme_crop,
                     augment_proxy_representation_canny,
                     augment_rgb)

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


def canny_edge_silhouette(seg, resized_img, target_joints2d_coco, background, edge_detect_model):
    # Add background rgb
    rgb_in = batch_add_rgb_background(backgrounds=background,
                                        rgb=resized_img,
                                        seg=seg)
    edge_detector_output = edge_detect_model(rgb_in)
    edge_in = edge_detector_output['thresholded_thin_edges'] if config.EDGE_NMS else edge_detector_output['thresholded_grad_magnitude']
    j2d_heatmaps = convert_2Djoints_to_gaussian_heatmaps_torch(target_joints2d_coco,
                                                                config.REGRESSOR_IMG_WH)
    proxy_rep_input = torch.cat([edge_in, j2d_heatmaps], dim=1).float()  # (batch_size, C, img_wh, img_wh)
    return proxy_rep_input


def canny_edge_augment_silhouette(resized_input, target_joints2d_coco, resized_img, target_joints2d_visib_coco, proxy_rep_augment_params, background, edge_detect_model):
    segmentation_map, target_joints2d_coco_input, target_joints2d_visib_coco = augment_proxy_representation_canny(
                                            seg=resized_input,  # Note: out of frame pixels marked with -1
                                            joints2D=target_joints2d_coco,
                                            joints2D_visib=target_joints2d_visib_coco,
                                            proxy_rep_augment_params=proxy_rep_augment_params)
    # Add background rgb
    rgb_in = batch_add_rgb_background(backgrounds=background,
                                        rgb=resized_img,
                                        seg=segmentation_map)
    # Apply RGB-based render augmentations + 2D joints augmentations
    rgb_in, target_joints2d_coco_input, target_joints2d_visib_coco = augment_rgb(rgb=rgb_in,
                                                                                    joints2D=target_joints2d_coco_input,
                                                                                    joints2D_visib=target_joints2d_visib_coco,
                                                                                    rgb_augment_config=config.AUGMENT_RGB)
    # Compute edge-images edges
    edge_detector_output = edge_detect_model(rgb_in)
    edge_in = edge_detector_output['thresholded_thin_edges'] if config.EDGE_NMS else edge_detector_output['thresholded_grad_magnitude']
    # Compute 2D joint heatmaps
    j2d_heatmaps = convert_2Djoints_to_gaussian_heatmaps_torch(target_joints2d_coco_input,
                                                                config.REGRESSOR_IMG_WH)
    j2d_heatmaps = j2d_heatmaps * target_joints2d_visib_coco[:, :, None, None]

    # Concatenate edge-image and 2D joint heatmaps to create input proxy representation
    proxy_rep_input = torch.cat([edge_in, j2d_heatmaps], dim=1).float()  # (batch_size, C, img_wh, img_wh)
    return proxy_rep_input


def canny_edge_detector_processing(renderer_output, background, edge_detect_model, target_joints2d_coco, proxy_rep_augment_params, device, batch_view):
    target_joints2d_visib_coco = check_joints2d_visibility_torch(target_joints2d_coco, config.REGRESSOR_IMG_WH)
    rgb_in = renderer_output['rgb_images'].permute(0, 3, 1, 2).contiguous()  # (bs, 3, img_wh, img_wh)
    iuv_in = renderer_output['iuv_images'].permute(0, 3, 1, 2).contiguous()  # (bs, 3, img_wh, img_wh)
    iuv_in[:, 1:, :, :] = iuv_in[:, 1:, :, :] * 255
    seg_map = iuv_in[:, 0, :, :].round()
    if batch_view:
        seg_map = random_extreme_crop(seg=seg_map, extreme_crop_probability=config.EXTREME_CROP_PROB)

    input_gt = convert_densepose_to_6part_lsp_labels(seg_map.cpu().numpy().astype(np.uint8))
    # Crop to person bounding box after bbox scale and centre augmentation
    if config.CROP_INPUT:
        # Crop inputs according to bounding box
        # + add random scale and centre augmentation
        target_joints2d_coco = target_joints2d_coco.cpu().detach().numpy()
        all_cropped_segs, all_cropped_joints2D, all_cropped_img = batch_crop_seg_to_bounding_box(
            input_gt, target_joints2d_coco, rgb_in,
            orig_scale_factor=config.MEAN_SCALE_FACTOR,
            delta_scale_range=config.DELTA_SCALE_RANGE,
            delta_centre_range=config.DELTA_CENTRE_RANGE)
        resized_input, resized_joints2D, resized_img = batch_resize(all_cropped_segs, all_cropped_joints2D, all_cropped_img, config.REGRESSOR_IMG_WH)
        resized_input = torch.from_numpy(resized_input).float().to(device)
        target_joints2d_coco = torch.from_numpy(resized_joints2D).float().to(device)
        resized_img = resized_img.to(device)

    if batch_view:
        input_representation = canny_edge_augment_silhouette(resized_input, target_joints2d_coco, resized_img, target_joints2d_visib_coco, proxy_rep_augment_params, background, edge_detect_model)
    else:
        input_representation = canny_edge_silhouette(seg=resized_input, 
                                                     resized_img=resized_img, 
                                                     target_joints2d_coco=target_joints2d_coco,
                                                     background=background,
                                                     edge_detect_model=edge_detect_model)
    return input_representation


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
                                            perspective_focal_length=config.FOCAL_LENGTH,
                                            render_rgb=True,
                                            bin_size=32)
        self.smpl = SMPL(config.SMPL_MODEL_DIR, batch_size=params['batch_size'])
        self.smpl.to(self.device)
        self.canny_edge_detctor = CannyEdgeDetector()
        self.canny_edge_detctor.to(self.device)
        self.lights_rgb_settings = {'location': torch.tensor([[0., -0.8, -2.0]], device=self.device, dtype=torch.float32),
                           'ambient_color': 0.5 * torch.ones(1, 3, device=self.device, dtype=torch.float32),
                           'diffuse_color': 0.3 * torch.ones(1, 3, device=self.device, dtype=torch.float32),
                           'specular_color': torch.zeros(1, 3, device=self.device, dtype=torch.float32)}
        self.proxy_rep_augment_params = {'remove_appendages': config.REMOVE_APPENDAGES,
                                'deviate_joints2D': config.DEVIATE_JOINTS2D,
                                'deviate_verts2D': config.DEVIATE_VERTS2D,
                                'occlude_seg': config.OCCLUDE_SEG,
                                'remove_appendages_classes': config.REMOVE_APPENDAGES_CLASSES,
                                'remove_appendages_probabilities': config.REMOVE_APPENDAGES_PROBABILITIES,
                                'delta_j2d_dev_range': config.DELTA_J2D_DEV_RANGE,
                                'delta_j2d_hip_dev_range': config.DELTA_J2D_HIP_DEV_RANGE,
                                'delta_verts2d_dev_range': config.DELTA_VERTS2D_DEV_RANGE,
                                'occlude_probability': config.OCCLUDE_PROBABILITY,
                                'occlude_box_dim': config.OCCLUDE_BOX_DIM,
                                'random_remove_joints': config.REMOVE_JOINTS,
                                'remove_joints_indices':config.REMOVE_JOINTS_INDICES,
                                'remove_joints_prob': config.REMOVE_JOINTS_PROB}


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

    @torch.no_grad()
    def process_batch_view(self, samples, second_view=False):
        target_pose = samples['pose']
        target_shape = samples['shape']
        target_pose = target_pose.to(self.device)
        target_shape = target_shape.to(self.device)
        background = samples['background'].to(self.device)  # (bs, 3, img_wh, img_wh)
        texture = samples['texture'].to(self.device)  # (bs, 1200, 800, 3)
        target_smpl_output = self.smpl(body_pose=target_pose[:, 3:],
                                       global_orient=target_pose[:, :3],
                                       betas=target_shape)
        target_vertices = target_smpl_output.vertices
        target_joints_all = target_smpl_output.joints
        target_joints_coco = target_joints_all[:, config.ALL_JOINTS_TO_COCO_MAP, :]
        target_joints2d_coco = perspective_project_torch(target_joints_coco, self.cam_R,
                                                         self.mean_cam_t,
                                                         cam_K=self.cam_K)
        # TARGET VERTICES AND JOINTS
        target_smpl_output = self.smpl(body_pose=target_pose[:, 3:],
                                        global_orient=target_pose[:, :3],
                                        betas=target_shape)
        
        target_vertices = target_smpl_output.vertices
        target_joints_all = target_smpl_output.joints
        target_joints_h36m = target_joints_all[:, config.ALL_JOINTS_TO_H36M_MAP, :]
        target_joints_h36mlsp = target_joints_h36m[:, config.H36M_TO_J14, :]
        target_joints_coco = target_joints_all[:, config.ALL_JOINTS_TO_COCO_MAP, :]
        target_joints2d_coco = perspective_project_torch(target_joints_coco, self.cam_R,
                                                            self.mean_cam_t,
                                                            cam_K=self.cam_K)
        # INPUT PROXY REPRESENTATION GENERATION
        renderer_output = self.pytorch3d_renderer(vertices=target_vertices,
                                    textures=texture,
                                    cam_t=self.mean_cam_t,
                                    lights_rgb_settings=self.lights_rgb_settings)
        input_repr = canny_edge_detector_processing(renderer_output, background, 
                                                    self.canny_edge_detctor, 
                                                    target_joints2d_coco,
                                                    self.proxy_rep_augment_params,
                                                    self.device,
                                                    second_view
                                                    )
        return input_repr


    def train(self, train_dataset):

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                  num_workers=self.num_workers, drop_last=True, shuffle=True)

        niter = 0
        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')

        self.initializes_target_network()

        for epoch_counter in range(self.max_epochs):

            for (batch_view_1, batch_view_2) in train_loader:

                with torch.no_grad():
                    batch_view_1 = self.process_batch_view(batch_view_1)
                    batch_view_2 = self.process_batch_view(batch_view_2, second_view=True)

                if niter == 0:
                    grid = torchvision.utils.make_grid(batch_view_1[:32, 0,:,:].unsqueeze(1))
                    self.writer.add_image('views_1', grid, global_step=niter)

                    grid = torchvision.utils.make_grid(batch_view_2[:32, 0,:,:].unsqueeze(1))
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
