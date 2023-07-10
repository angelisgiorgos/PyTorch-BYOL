import torch
import config
from utils3d.label_conversions import (
    convert_densepose_to_6part_lsp_labels,
    convert_multiclass_to_binary_labels, convert_2Djoints_to_gaussian_heatmaps_torch
    )
from utils3d.image_utils import batch_crop_seg_to_bounding_box, batch_resize
from utils3d.proxy_repr_augmentation import augment_proxy_representation

proxy_rep_augment_params = {
    'remove_appendages': config.REMOVE_APPENDAGES,
    'deviate_joints2D': config.DEVIATE_JOINTS2D,
    'deviate_verts2D': config.DEVIATE_VERTS2D,
    'occlude_seg': config.OCCLUDE_SEG,
    'remove_appendages_classes': config.REMOVE_APPENDAGES_CLASSES,
    'remove_appendages_probabilities': config.REMOVE_APPENDAGES_PROBABILITIES,
    'delta_j2d_dev_range': config.DELTA_J2D_DEV_RANGE,
    'delta_j2d_hip_dev_range': config.DELTA_J2D_HIP_DEV_RANGE,
    'delta_verts2d_dev_range': config.DELTA_VERTS2D_DEV_RANGE,
    'occlude_probability': config.OCCLUDE_PROBABILITY,
    'occlude_box_dim': config.OCCLUDE_BOX_DIM}


def smpl_based_val_representation(config, target_vertices, target_joints2d_coco, renderer, cam_t,
                                  device, second_view=False):
    renderer_output = renderer(vertices=target_vertices,
                               cam_t=cam_t,
                               lights_rgb_settings=None)
    iuv_in = renderer_output['iuv_images'].permute(0, 3, 1,
                                                   2).contiguous()  # (bs, 3, img_wh, img_wh)
    input_gt = iuv_in[:, 0, :, :].round()
    input_gt = convert_densepose_to_6part_lsp_labels(input_gt.long())
    if config.CROP_INPUT:
        target_joints2d_coco = target_joints2d_coco.cpu().detach().numpy()
        all_cropped_segs, all_cropped_joints2D = batch_crop_seg_to_bounding_box(
            input_gt, target_joints2d_coco,
            orig_scale_factor=config.MEAN_SCALE_FACTOR,
            delta_scale_range=config.DELTA_SCALE_RANGE,
            delta_centre_range=config.DELTA_CENTRE_RANGE)
        resized_input, resized_joints2D = batch_resize(all_cropped_segs, all_cropped_joints2D,
                                                       config.REGRESSOR_IMG_WH)
        resized_input = torch.from_numpy(resized_input).float().to(device)
        target_joints2d_coco = torch.from_numpy(resized_joints2D).float().to(device)

    if second_view:
        resized_input, target_joints2d_coco = augment_proxy_representation(resized_input,
                                                                           target_joints2d_coco,
                                                                           proxy_rep_augment_params)
    # FINAL INPUT PROXY REPRESENTATION GENERATION WITH JOINT HEATMAPS
    segmentation_map = convert_multiclass_to_binary_labels(resized_input)
    segmentation_map = segmentation_map.unsqueeze(1)
    j2d_heatmaps = convert_2Djoints_to_gaussian_heatmaps_torch(target_joints2d_coco,
                                                               config.REGRESSOR_IMG_WH)
    final_repr = torch.cat([segmentation_map, j2d_heatmaps], dim=1)
    return final_repr
