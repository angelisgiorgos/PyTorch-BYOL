import torch
import numpy as np
from utils3d.label_conversions import convert_densepose_to_6part_lsp_labels, convert_multiclass_to_binary_labels, convert_2Djoints_to_gaussian_heatmaps_torch
from utils3d.image_utils import batch_crop_seg_to_bounding_box_torch, batch_resize_torch

def smpl_based_val_representation(config, target_vertices, target_joints2d_coco, renderer, cam_t, device): 
    
    renderer_output = renderer(vertices=target_vertices,
                                cam_t=cam_t,
                                lights_rgb_settings=None)
    iuv_in = renderer_output['iuv_images'].permute(0, 3, 1, 2).contiguous()  # (bs, 3, img_wh, img_wh)
    input_gt = iuv_in[:, 0, :, :].round()
    input_gt = convert_densepose_to_6part_lsp_labels(input_gt.long())
    if config.CROP_INPUT:
        all_cropped_segs, all_cropped_joints2D = batch_crop_seg_to_bounding_box_torch(
            input_gt, target_joints2d_coco,
            orig_scale_factor=config.MEAN_SCALE_FACTOR,
            delta_scale_range=None,
            delta_centre_range=None, 
            device = device)
    input_representation = convert_multiclass_to_binary_labels(all_cropped_segs)
    j2d_heatmaps = convert_2Djoints_to_gaussian_heatmaps_torch(all_cropped_joints2D,
                                                                config.REGRESSOR_IMG_WH)
    final_repr = torch.cat([input_representation.unsqueeze(1), j2d_heatmaps], dim=1)
    return final_repr

