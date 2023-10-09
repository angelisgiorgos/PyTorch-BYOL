import torch.nn.functional as F
import os
import cv2
import torch
import numpy as np
import skimage
from PIL import Image
import config
import torchvision

def pad_to_square(image):
    """
    Pad image to square shape.
    """
    height, width = image.shape[:2]

    if width < height:
        border_width = (height - width) // 2
        image = cv2.copyMakeBorder(image, 0, 0, border_width, border_width,
                                   cv2.BORDER_CONSTANT, value=0)
    else:
        border_width = (width - height) // 2
        image = cv2.copyMakeBorder(image, border_width, border_width, 0, 0,
                                   cv2.BORDER_CONSTANT, value=0)

    return image

def convert_bbox_corners_to_centre_hw(bbox_corners):
    """
    Converst bbox coordinates from x1, y1, x2, y2 to centre, height, width.
    """
    x1, y1, x2, y2 = bbox_corners
    centre = np.array([(x1+x2)/2.0, (y1+y2)/2.0])
    height = x2 - x1
    width = y2 - y1

    return centre, height, width


def convert_bbox_corners_to_centre_hw_torch(bbox_corners,dev):
    """
    Converst bbox coordinates from x1, y1, x2, y2 to centre, height, width.
    """
    x1, y1, x2, y2 = bbox_corners
    centre = torch.tensor([(x1+x2)/2.0, (y1+y2)/2.0],device=dev)
    height = x2 - x1
    width = y2 - y1

    return centre, height, width


def convert_bbox_centre_hw_to_corners(centre, height, width):
    x1 = centre[0] - height/2.0
    x2 = centre[0] + height/2.0
    y1 = centre[1] - width/2.0
    y2 = centre[1] + width/2.0

    return np.array([x1, y1, x2, y2])


def convert_bbox_centre_hw_to_corners_troch(centre, height, width):
    x1 = centre[0] - height/2.0
    x2 = centre[0] + height/2.0
    y1 = centre[1] - width/2.0
    y2 = centre[1] + width/2.0

    return torch.tensor([x1, y1, x2, y2])    


def batch_crop_seg_to_bounding_box(seg, joints2D, img=None, orig_scale_factor=1.2, delta_scale_range=None, delta_centre_range=None):
    """
    seg: (bs, wh, wh)
    joints2D: (bs, num joints, 2)
    scale: bbox expansion scale
    """
    all_cropped_segs = []
    all_cropped_joints2D = []
    all_cropped_imgs = []
    for i in range(seg.shape[0]):
        body_pixels = np.argwhere(seg[i] != 0)
        bbox_corners = np.amin(body_pixels, axis=0), np.amax(body_pixels, axis=0)
        bbox_corners = np.concatenate(bbox_corners)
        centre, height, width = convert_bbox_corners_to_centre_hw(bbox_corners)
        if delta_scale_range is not None:
            l, h = delta_scale_range
            delta_scale = (h - l) * np.random.rand() + l
            scale_factor = orig_scale_factor + delta_scale
        else:
            scale_factor = orig_scale_factor

        if delta_centre_range is not None:
            l, h = delta_centre_range
            delta_centre = (h - l) * np.random.rand(2) + l
            centre = centre + delta_centre

        wh = max(height, width) * scale_factor

        bbox_corners = convert_bbox_centre_hw_to_corners(centre, wh, wh)

        top_left = bbox_corners[:2].astype(np.int16)
        bottom_right = bbox_corners[2:].astype(np.int16)
        top_left[top_left < 0] = 0
        bottom_right[bottom_right < 0] = 0

        cropped_joints2d = joints2D[i] - top_left[::-1]
        cropped_seg = seg[i, top_left[0]: bottom_right[0], top_left[1]: bottom_right[1]]
        all_cropped_joints2D.append(cropped_joints2d)
        all_cropped_segs.append(cropped_seg)
        if img is not None:
            cropped_img = img[i, :, top_left[0]: bottom_right[0], top_left[1]: bottom_right[1]]
            all_cropped_imgs.append(cropped_img)

    return all_cropped_segs, all_cropped_joints2D, all_cropped_imgs


def batch_resize(all_cropped_segs, all_cropped_joints2D, all_cropped_images=None, img_wh=256):
    """
    all_cropped_seg: list of cropped segs with len = batch size
    """
    all_resized_segs = []
    all_resized_joints2D = []
    all_resized_imgs = []
    for i in range(len(all_cropped_segs)):
        seg = all_cropped_segs[i]
        orig_height, orig_width = seg.shape[:2]
        resized_seg = cv2.resize(seg, (img_wh, img_wh), interpolation=cv2.INTER_NEAREST)
        all_resized_segs.append(resized_seg)

        joints2D = all_cropped_joints2D[i]
        resized_joints2D = joints2D * np.array([img_wh / float(orig_width),
                                                img_wh / float(orig_height)])
        all_resized_joints2D.append(resized_joints2D)
        if all_cropped_images:
            transform = torchvision.transforms.Resize((img_wh, img_wh))
            transform_iamges = transform(all_cropped_images[i])
            all_resized_imgs.append(transform_iamges)


    all_resized_segs = np.stack(all_resized_segs, axis=0)
    all_resized_joints2D = np.stack(all_resized_joints2D, axis=0)
    if all_cropped_images:
        all_resized_imgs = torch.stack(all_resized_imgs, axis=0)


    return all_resized_segs, all_resized_joints2D, all_resized_imgs



def crop_and_resize_silhouette_joints(silhouette,
                                      joints2D,
                                      out_wh,
                                      image=None,
                                      image_out_wh=None,
                                      bbox_scale_factor=1.2):
    # Find bounding box around silhouette
    body_pixels = np.argwhere(silhouette != 0)
    bbox_centre, height, width = convert_bbox_corners_to_centre_hw(np.concatenate([np.amin(body_pixels, axis=0),
                                                                                   np.amax(body_pixels, axis=0)]))
    wh = max(height, width) * bbox_scale_factor  # Make bounding box square with sides = wh
    bbox_corners = convert_bbox_centre_hw_to_corners(bbox_centre, wh, wh)
    top_left = bbox_corners[:2].astype(np.int16)
    bottom_right = bbox_corners[2:].astype(np.int16)
    top_left_orig = top_left.copy()
    bottom_right_orig = bottom_right.copy()
    top_left[top_left < 0] = 0
    bottom_right[bottom_right < 0] = 0
    # Crop silhouette
    orig_height, orig_width = silhouette.shape[:2]
    silhouette = silhouette[top_left[0]: bottom_right[0], top_left[1]: bottom_right[1]]
    # Pad silhouette if crop not square
    silhouette = cv2.copyMakeBorder(src=silhouette,
                                    top=max(0, -top_left_orig[0]),
                                    bottom=max(0, bottom_right_orig[0] - orig_height),
                                    left=max(0, -top_left_orig[1]),
                                    right=max(0, bottom_right_orig[1] - orig_width),
                                    borderType=cv2.BORDER_CONSTANT,
                                    value=0)
    crop_height, crop_width = silhouette.shape[:2]
    # Resize silhouette
    silhouette = cv2.resize(silhouette, (out_wh, out_wh),
                            interpolation=cv2.INTER_NEAREST)

    # Translate and resize joints2D
    joints2D = joints2D[:, :2] - top_left_orig[::-1]
    joints2D = joints2D * np.array([out_wh / float(crop_width),
                                    out_wh / float(crop_height)])

    if image is not None:
        # Crop image
        orig_height, orig_width = image.shape[:2]
        image = image[top_left[0]: bottom_right[0], top_left[1]: bottom_right[1]]
        # Pad image if crop not square
        image = cv2.copyMakeBorder(src=image,
                                   top=max(0, -top_left_orig[0]),
                                   bottom=max(0, bottom_right_orig[0] - orig_height),
                                   left=max(0, -top_left_orig[1]),
                                   right=max(0, bottom_right_orig[1] - orig_width),
                                   borderType=cv2.BORDER_CONSTANT,
                                   value=0)
        # Resize silhouette
        image = cv2.resize(image, (image_out_wh, image_out_wh),
                           interpolation=cv2.INTER_LINEAR)

    return silhouette, joints2D, image


def batch_crop_seg_to_bounding_box_torch(seg, joints2D, img=None, orig_scale_factor=1.2, delta_scale_range=None, delta_centre_range=None, device=None):
    """
    seg: (bs, wh, wh)
    joints2D: (bs, num joints, 2)
    scale: bbox expansion scale
    """
    all_cropped_segs = []
    all_cropped_joints2D = []
    all_cropped_imgs = []
    for i in range(seg.shape[0]):
        body_pixels =  torch.nonzero(seg[i])
        bbox_corners = torch.amin(body_pixels, axis=0), torch.amax(body_pixels, axis=0)
        bbox_corners = torch.cat(bbox_corners)
        centre, height, width = convert_bbox_corners_to_centre_hw_torch(bbox_corners, device)

        if delta_scale_range is not None:
            l, h = delta_scale_range
            delta_scale = (h - l) * torch.rand(1,device=device) + l
            scale_factor = orig_scale_factor + delta_scale
        else:
            scale_factor = orig_scale_factor
       
       
        if delta_centre_range is not None:
            l, h = delta_centre_range
            delta_centre = (h - l) *torch.rand(2, device=device) + l
            centre = centre + delta_centre  
            

        wh = torch.max(height, width) * scale_factor
        bbox_corners = convert_bbox_centre_hw_to_corners_troch(centre, wh, wh)
        top_left = bbox_corners[:2].type(torch.cuda.ShortTensor)
        bottom_right = bbox_corners[2:].type(torch.cuda.ShortTensor)
        top_left[top_left < 0] = 0
        bottom_right[bottom_right < 0] = 0

        cropped_joints2d = joints2D[i] - torch.flip(top_left,[0])
        cropped_seg = seg[i, top_left[0]: bottom_right[0], top_left[1]: bottom_right[1]]
        all_cropped_joints2D.append(cropped_joints2d)
        all_cropped_segs.append(cropped_seg)
        if img is not None:
            cropped_img = img[i, :, top_left[0]: bottom_right[0], top_left[1]: bottom_right[1]]
            all_cropped_imgs.append(cropped_img)
            
    segmentations, joints2D = batch_resize_torch(all_cropped_segs, all_cropped_joints2D)

    return segmentations, joints2D


def batch_resize_torch(all_cropped_segs, all_cropped_joints2D, img_wh=256):
    """
    all_cropped_seg: list of cropped segs with len = batch size
    """
    all_resized_segs = []
    all_resized_joints2D = []
    
    dev = 'cuda:' +str(all_cropped_segs[0].get_device())
    #print('device inside batch corp', dev)
    for i in range(len(all_cropped_segs)):
        seg = all_cropped_segs[i]
        orig_height, orig_width = seg.shape[:2]
        if len(seg.shape) == 2:
            seg = seg.unsqueeze(0).unsqueeze(0)
        else:
            seg = seg.unsqueeze(0)
        resized_seg = torch.nn.functional.interpolate(seg.float(), size = (img_wh, img_wh))
        resized_seg = resized_seg.squeeze(0)
        resized_seg = resized_seg.squeeze(0)

        all_resized_segs.append(resized_seg)

        joints2D = all_cropped_joints2D[i]
        resized_joints2D = joints2D * torch.tensor([img_wh / float(orig_width),img_wh / float(orig_height)],device=dev)
        all_resized_joints2D.append(resized_joints2D)

    all_resized_segs = torch.stack(all_resized_segs, axis=0)
    all_resized_joints2D = torch.stack(all_resized_joints2D, axis=0)
    return all_resized_segs, all_resized_joints2D



def batch_add_rgb_background(backgrounds,
                             rgb,
                             seg):
    """
    :param backgrounds: (bs, 3, wh, wh)
    :param rgb: (bs, 3, wh, wh)
    :param iuv: (bs, wh, wh)
    :return: rgb_with_background: (bs, 3, wh, wh)
    """
    background_pixels = seg[:, None, :, :] == 0  # Body pixels are > 0 and out of frame pixels are -1
    rgb_with_background = rgb * (torch.logical_not(background_pixels)) + backgrounds * background_pixels
    return rgb_with_background



def batch_crop_opencv_affine(output_wh,
                             num_to_crop,
                             iuv=None,
                             joints2D=None,
                             rgb=None,
                             seg=None,
                             bbox_centres=None,
                             bbox_heights=None,
                             bbox_widths=None,
                             bbox_whs=None,
                             joints2D_vis=None,
                             orig_scale_factor=1.2,
                             delta_scale_range=None,
                             delta_centre_range=None,
                             out_of_frame_pad_val=0,
                             solve_for_affine_trans=False,
                             uncrop=False,
                             uncrop_wh=None):
    """
    :param output_wh: tuple, output image (width, height)
    :param num_to_crop: scalar int, number of images in batch
    :param iuv: (B, 3, H, W)
    :param joints2D: (B, K, 2)
    :param rgb: (B, 3, H, W)
    :param seg: (B, H, W)
    :param bbox_centres: (B, 2), bounding box centres in (vertical, horizontal) coordinates
    :param bbox_heights: (B,)
    :param bbox_widths: (B,)
    :param bbox_whs: (B,) width/height for square bounding boxes
    :param joints2D_vis: (B, K)
    :param orig_scale_factor: original bbox scale factor (pre-augmentation)
    :param delta_scale_range: bbox scale augmentation range
    :param delta_centre_range: bbox centre augmentation range
    :param out_of_frame_pad_val: padding value for out-of-frame region after affine transform
    :param solve_for_affine_trans: bool, if true use cv2.getAffineTransform() to determine
        affine transformation matrix.
    :param uncrop: bool, if true uncrop image by applying inverse affine transformation
    :param uncrop_wh: tuple, output image size for uncropping.
    :return: cropped iuv/joints2D/rgb/seg, resized to output_wh
    """
    output_wh = np.array(output_wh, dtype=np.float32)
    cropped_dict = {}
    if iuv is not None:
        if not uncrop:
            cropped_dict['iuv'] = np.zeros((iuv.shape[0], 3, int(output_wh[1]), int(output_wh[0])), dtype=np.float32)
        else:
            cropped_dict['iuv'] = np.zeros((iuv.shape[0], 3, int(uncrop_wh[1]), int(uncrop_wh[0])), dtype=np.float32)
    if joints2D is not None:
        cropped_dict['joints2D'] = np.zeros_like(joints2D)
    if rgb is not None:
        if not uncrop:
            cropped_dict['rgb'] = np.zeros((rgb.shape[0], 3, int(output_wh[1]), int(output_wh[0])), dtype=np.float32)
        else:
            cropped_dict['rgb'] = np.zeros((rgb.shape[0], 3, int(uncrop_wh[1]), int(uncrop_wh[0])), dtype=np.float32)
    if seg is not None:
        if not uncrop:
            cropped_dict['seg'] = np.zeros((seg.shape[0], int(output_wh[1]), int(output_wh[0])), dtype=np.float32)
        else:
            cropped_dict['seg'] = np.zeros((seg.shape[0], int(uncrop_wh[1]), int(uncrop_wh[0])), dtype=np.float32)

    for i in range(num_to_crop):
        if bbox_centres is None:
            assert (iuv is not None) or (joints2D is not None) or (seg is not None), "Need either IUV, Seg or 2D Joints to determine bounding boxes!"
            if iuv is not None:
                # Determine bounding box corners from segmentation foreground/body pixels from IUV map
                body_pixels = np.argwhere(iuv[i, 0, :, :] != 0)
                bbox_corners = np.concatenate([np.amin(body_pixels, axis=0),
                                               np.amax(body_pixels, axis=0)])
            elif seg is not None:
                # Determine bounding box corners from segmentation foreground/body pixels
                body_pixels = np.argwhere(seg[i] != 0)
                bbox_corners = np.concatenate([np.amin(body_pixels, axis=0),
                                               np.amax(body_pixels, axis=0)])
            elif joints2D is not None:
                # Determine bounding box corners from 2D joints
                visible_joints2D = joints2D[i, joints2D_vis[i]]
                bbox_corners = np.concatenate([np.amin(visible_joints2D, axis=0)[::-1],   # (hor, vert) coords to (vert, hor) coords
                                               np.amax(visible_joints2D, axis=0)[::-1]])
                if (bbox_corners[:2] == bbox_corners[2:]).all():  # This can happen if only 1 joint is visible in input
                    print('Only 1 visible joint in input!')
                    bbox_corners[2:] = bbox_corners[:2] + output_wh / 4.
            bbox_centre, bbox_height, bbox_width = convert_bbox_corners_to_centre_hw(bbox_corners)
        else:
            bbox_centre = bbox_centres[i]
            if bbox_whs is not None:
                bbox_height = bbox_whs[i]
                bbox_width = bbox_whs[i]
            else:
                bbox_height = bbox_heights[i]
                bbox_width = bbox_widths[i]

        if not uncrop:
            # Change bounding box aspect ratio to match output aspect ratio
            aspect_ratio = output_wh[1] / output_wh[0]
            if bbox_height > bbox_width * aspect_ratio:
                bbox_width = bbox_height / aspect_ratio
            elif bbox_height < bbox_width * aspect_ratio:
                bbox_height = bbox_width * aspect_ratio

            # Scale bounding boxes + Apply random augmentations
            if delta_scale_range is not None:
                l, h = delta_scale_range
                delta_scale = (h - l) * np.random.rand() + l
                scale_factor = orig_scale_factor + delta_scale
            else:
                scale_factor = orig_scale_factor
            bbox_height = bbox_height * scale_factor
            bbox_width = bbox_width * scale_factor
            if delta_centre_range is not None:
                l, h = delta_centre_range
                delta_centre = (h - l) * np.random.rand(2) + l
                bbox_centre = bbox_centre + delta_centre

            # Determine affine transformation mapping bounding box to output image
            output_centre = output_wh * 0.5
            if solve_for_affine_trans:
                # Solve for affine transformation using 3 point correspondences (6 equations)
                bbox_points = np.zeros((3, 2), dtype=np.float32)
                bbox_points[0, :] = bbox_centre[::-1]  # (vert, hor) coordinates to (hor, vert coordinates)
                bbox_points[1, :] = bbox_centre[::-1] + np.array([bbox_width * 0.5, 0], dtype=np.float32)
                bbox_points[2, :] = bbox_centre[::-1] + np.array([0, bbox_height * 0.5], dtype=np.float32)

                output_points = np.zeros((3, 2), dtype=np.float32)
                output_points[0, :] = output_centre
                output_points[1, :] = output_centre + np.array([output_wh[0] * 0.5, 0], dtype=np.float32)
                output_points[2, :] = output_centre + np.array([0, output_wh[1] * 0.5], dtype=np.float32)
                affine_trans = cv2.getAffineTransform(bbox_points, output_points)
            else:
                # Hand-code affine transformation matrix - easy for cropping = scale + translate
                affine_trans = np.zeros((2, 3), dtype=np.float32)
                affine_trans[0, 0] = output_wh[0] / bbox_width
                affine_trans[1, 1] = output_wh[1] / bbox_height
                affine_trans[:, 2] = output_centre - (output_wh / np.array([bbox_width, bbox_height])) * bbox_centre[::-1]  # (vert, hor) coords to (hor, vert) coords
        else:
            # Hand-code inverse affine transformation matrix - easy for UN-cropping = scale + translate
            affine_trans = np.zeros((2, 3), dtype=np.float32)
            output_centre = output_wh * 0.5
            affine_trans[0, 0] = bbox_width / output_wh[0]
            affine_trans[1, 1] = bbox_height / output_wh[1]
            affine_trans[:, 2] = bbox_centre[::-1] - (np.array([bbox_width, bbox_height]) / output_wh) * output_centre

        # Apply affine transformation inputs.
        if iuv is not None:
            cropped_dict['iuv'][i] = cv2.warpAffine(src=iuv[i].transpose(1, 2, 0),
                                                    M=affine_trans,
                                                    dsize=tuple(output_wh.astype(np.int16)) if not uncrop else uncrop_wh,
                                                    flags=cv2.INTER_NEAREST,
                                                    borderMode=cv2.BORDER_CONSTANT,
                                                    borderValue=out_of_frame_pad_val).transpose(2, 0, 1)
        if joints2D is not None:
            joints2D_homo = np.concatenate([joints2D[i], np.ones((joints2D.shape[1], 1))],
                                           axis=-1)
            cropped_dict['joints2D'][i] = np.einsum('ij,kj->ki', affine_trans, joints2D_homo)

        if rgb is not None:
            cropped_dict['rgb'][i] = cv2.warpAffine(src=rgb[i].transpose(1, 2, 0),
                                                    M=affine_trans,
                                                    dsize=tuple(output_wh.astype(np.int16)) if not uncrop else uncrop_wh,
                                                    flags=cv2.INTER_LINEAR,
                                                    borderMode=cv2.BORDER_CONSTANT,
                                                    borderValue=0).transpose(2, 0, 1)
        if seg is not None:
            cropped_dict['seg'][i] = cv2.warpAffine(src=seg[i],
                                                    M=affine_trans,
                                                    dsize=tuple(output_wh.astype(np.int16)) if not uncrop else uncrop_wh,
                                                    flags=cv2.INTER_NEAREST,
                                                    borderMode=cv2.BORDER_CONSTANT,
                                                    borderValue=0)

    return cropped_dict


def get_transform(center, scale, res, rot=0):
    """Generate transformation matrix."""
    h = 200 * scale
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / h
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / h + .5)
    t[1, 2] = res[0] * (-float(center[1]) / h + .5)
    t[2, 2] = 1
    if not rot == 0:
        rot = -rot # To match direction of rotation from cropping
        rot_mat = np.zeros((3,3))
        rot_rad = rot * np.pi / 180
        sn,cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0,:2] = [cs, -sn]
        rot_mat[1,:2] = [sn, cs]
        rot_mat[2,2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0,2] = -res[1]/2
        t_mat[1,2] = -res[0]/2
        t_inv = t_mat.copy()
        t_inv[:2,2] *= -1
        t = np.dot(t_inv,np.dot(rot_mat,np.dot(t_mat,t)))
    return t



def transform(pt, center, scale, res, invert=0, rot=0):
    """Transform pixel location to different reference."""
    t = get_transform(center, scale, res, rot=rot)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array([pt[0] - 1, pt[1] - 1, 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2].astype(int) + 1

def crop(img, center, scale, res, rot=0):
    """Crop image according to the supplied bounding box."""
    # Upper left point
    ul = np.array(transform([1, 1], center, scale, res, invert=1))-1
    # Bottom right point
    br = np.array(transform([res[0]+1, 
                             res[1]+1], center, scale, res, invert=1))-1
    
    # Padding so that when rotated proper amount of context is included
    pad = int(np.linalg.norm(br - ul) / 2 - float(br[1] - ul[1]) / 2)
    if not rot == 0:
        ul -= pad
        br += pad

    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    if len(img.shape) > 2:
        new_shape += [img.shape[2]]
    new_img = np.zeros(new_shape, dtype=np.float32)

    # Range to fill new array
    new_x = max(0, -ul[0]), min(br[0], len(img[0])) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], len(img)) - ul[1]
    # Range to sample from original image
    old_x = max(0, ul[0]), min(len(img[0]), br[0])
    old_y = max(0, ul[1]), min(len(img), br[1])
    new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1], old_x[0]:old_x[1]]

    if not rot == 0:
        # Remove padding
        new_img = skimage.transform.rotate(new_img, rot).astype(np.uint8)
        new_img = new_img[pad:-pad, pad:-pad]

    new_img = np.array(Image.fromarray(new_img.astype(np.uint8)).resize(res))
    
    return new_img

def flip_img(img):
    """Flip rgb images or masks.
    channels come last, e.g. (256,256,3).
    """
    img = np.fliplr(img)
    return img

def flip_kp(kp):
    """Flip keypoints."""
    if len(kp) == 24:
        flipped_parts = config.J24_FLIP_PERM
    elif len(kp) == 49:
        flipped_parts = config.J49_FLIP_PERM
    kp = kp[flipped_parts]
    kp[:,0] = - kp[:,0]
    return kp

def flip_pose(pose):
    """Flip pose.
    The flipping is based on SMPL parameters.
    """
    flipped_parts = config.SMPL_POSE_FLIP_PERM
    pose = pose[flipped_parts]
    # we also negate the second and the third dimension of the axis-angle
    pose[1::3] = -pose[1::3]
    pose[2::3] = -pose[2::3]
    return pose



def batch_crop_pytorch_affine(input_wh,
                              output_wh,
                              num_to_crop,
                              device,
                              iuv=None,
                              joints2D=None,
                              rgb=None,
                              seg=None,
                              bbox_determiner=None,
                              bbox_centres=None,
                              bbox_heights=None,
                              bbox_widths=None,
                              joints2D_vis=None,
                              orig_scale_factor=1.2,
                              delta_scale_range=None,
                              delta_centre_range=None,
                              out_of_frame_pad_val=0):
    """
    :param input_wh: tuple, input image (width, height)
    :param output_wh: tuple, output image (width, height)
    :param num_to_crop: number of images in batch
    :param iuv: (B, 3, H, W)
    :param joints2D: (B, K, 2)
    :param rgb: (B, 3, H, W)
    :param seg: (B, H, W)
    :param bbox_determiner: (B, H, W) segmentation/silhouette used to determine bbox corners if bbox corners
                            not determined by given iuv/joints2D/seg (e.g. used for extreme_crop augmentation)
    :param bbox_centres: (B, 2) bounding box centres in (vertical, horizontal) coordinates
    :param bbox_heights: (B,)
    :param bbox_widths: (B,
    :param joints2D_vis: (B, K)
    :param orig_scale_factor: original bbox scale factor (pre-augmentation)
    :param delta_scale_range: bbox scale augmentation range
    :param delta_centre_range: bbox centre augmentation range
    :param out_of_frame_pad_val: padding value for out-of-frame region after affine transform
    :return: Given iuv/joints2D/rgb/seg inputs, crops around person bounding box in input,
             resizes to output_wh and returns.
             Cropping + resizing is done using Pytorch's affine_grid and grid_sampling
    """
    input_wh = torch.tensor(input_wh, device=device, dtype=torch.float32)
    output_wh = torch.tensor(output_wh, device=device, dtype=torch.float32)
    if bbox_centres is None:
        # Need to determine bounding box from given IUV/Seg/2D Joints
        bbox_corners = torch.zeros(num_to_crop, 4, dtype=torch.float32, device=device)
        for i in range(num_to_crop):
            if bbox_determiner is None:
                assert (iuv is not None) or (joints2D is not None) or (seg is not None), "Need either IUV, Seg or 2D Joints to determine bounding boxes!"
                if iuv is not None:
                    # Determine bounding box corners from segmentation foreground/body pixels from IUV map
                    body_pixels = torch.nonzero(iuv[i, 0, :, :] != 0, as_tuple=False)
                    bbox_corners[i, :2], _ = torch.min(body_pixels, dim=0)  # Top left
                    bbox_corners[i, 2:], _ = torch.max(body_pixels, dim=0)  # Bot right
                elif seg is not None:
                    # Determine bounding box corners from segmentation foreground/body pixels
                    body_pixels = torch.nonzero(seg[i] != 0, as_tuple=False)
                    bbox_corners[i, :2], _ = torch.min(body_pixels, dim=0)  # Top left
                    bbox_corners[i, 2:], _ = torch.max(body_pixels, dim=0)  # Bot right
                elif joints2D is not None:
                    # Determine bounding box corners from 2D joints
                    visible_joints2D = joints2D[i, joints2D_vis[i]]
                    bbox_corners[i, :2], _ = torch.min(visible_joints2D, dim=0)  # Top left
                    bbox_corners[i, 2:], _ = torch.max(visible_joints2D, dim=0)  # Bot right
                    bbox_corners[i] = bbox_corners[i, [1, 0, 3, 2]]  # (horizontal, vertical) coordinates to (vertical, horizontal coordinates)
                    if (bbox_corners[:2] == bbox_corners[2:]).all():  # This can happen if only 1 joint is visible in input
                        print('Only 1 visible joint in input!')
                        bbox_corners[2] = bbox_corners[0] + output_wh[1]
                        bbox_corners[3] = bbox_corners[1] + output_wh[0]
            else:
                # Determine bounding box corners using given "bbox determiner"
                body_pixels = torch.nonzero(bbox_determiner[i] != 0, as_tuple=False)
                bbox_corners[i, :2], _ = torch.min(body_pixels, dim=0)  # Top left
                bbox_corners[i, 2:], _ = torch.max(body_pixels, dim=0)  # Bot right

        bbox_centres, bbox_heights, bbox_widths = convert_bbox_corners_to_centre_hw_torch(bbox_corners, device)

    # Change bounding box aspect ratio to match output aspect ratio
    aspect_ratio = (output_wh[1] / output_wh[0]).item()
    bbox_widths[bbox_heights > bbox_widths * aspect_ratio] = bbox_heights[bbox_heights > bbox_widths * aspect_ratio] / aspect_ratio
    bbox_heights[bbox_heights < bbox_widths * aspect_ratio] = bbox_widths[bbox_heights < bbox_widths * aspect_ratio] * aspect_ratio

    # Scale bounding boxes + Apply random augmentations
    if delta_scale_range is not None:
        l, h = delta_scale_range
        delta_scale = (h - l) * torch.rand(num_to_crop, device=device, dtype=torch.float32) + l
        scale_factor = orig_scale_factor + delta_scale
    else:
        scale_factor = orig_scale_factor
    bbox_heights = bbox_heights * scale_factor
    bbox_widths = bbox_widths * scale_factor
    if delta_centre_range is not None:
        l, h = delta_centre_range
        delta_centre = (h - l) * torch.rand(num_to_crop, 2, device=device, dtype=torch.float32) + l
        bbox_centres = bbox_centres + delta_centre

    # Hand-code affine transformation matrix - easy for cropping = scale + translate
    output_centre = output_wh * 0.5
    affine_trans = torch.zeros(num_to_crop, 2, 3, dtype=torch.float32, device=device)
    affine_trans[:, 0, 0] = output_wh[0] / bbox_widths
    affine_trans[:, 1, 1] = output_wh[1] / bbox_heights
    bbox_whs = torch.stack([bbox_widths, bbox_heights], dim=-1)
    affine_trans[:, :, 2] = output_centre - (output_wh / bbox_whs) * bbox_centres[:, [1, 0]]  # (vert, hor) to (hor, vert)

    # Pytorch needs NORMALISED INVERSE (compared to openCV) affine transform matrix for pytorch grid sampling
    # Since transform is just scale + translate, it is faster to hand-code noramlise + inverse than to use torch.inverse()
    # Forward transform: unnormalise with input dimensions + affine transform + normalise with output dimensions
    # Pytorch affine input = (Forward transform)^-1
    # see https://discuss.pytorch.org/t/affine-transformation-matrix-paramters-conversion/19522/18
    affine_trans_inv_normed = torch.zeros(num_to_crop, 2, 3, dtype=torch.float32, device=device)
    affine_trans_inv_normed[:, 0, 0] = bbox_widths / input_wh[0]
    affine_trans_inv_normed[:, 1, 1] = bbox_heights / input_wh[1]
    affine_trans_inv_normed[:, :, 2] = -affine_trans[:, :, 2] / (output_wh / bbox_whs)
    affine_trans_inv_normed[:, :, 2] = affine_trans_inv_normed[:, :, 2] / (input_wh * 0.5) + (bbox_whs / input_wh) - 1

    # Apply affine transformation inputs.
    affine_grid = F.affine_grid(theta=affine_trans_inv_normed,
                                size=[num_to_crop, 1, int(output_wh[1]), int(output_wh[0])],
                                align_corners=False)

    cropped_dict = {}
    if iuv is not None:
        cropped_dict['iuv'] = F.grid_sample(input=iuv - out_of_frame_pad_val,
                                            grid=affine_grid,
                                            mode='nearest',
                                            padding_mode='zeros',
                                            align_corners=False) + out_of_frame_pad_val
    if joints2D is not None:
        joints2D_homo = torch.cat([joints2D,
                                   torch.ones(num_to_crop, joints2D.shape[1], 1, device=device, dtype=torch.float32)],
                                  dim=-1)
        cropped_dict['joints2D'] = torch.einsum('bij,bkj->bki', affine_trans, joints2D_homo)

    if rgb is not None:
        cropped_dict['rgb'] = F.grid_sample(input=rgb,
                                            grid=affine_grid,
                                            mode='bilinear',
                                            padding_mode='zeros',
                                            align_corners=False)
    if seg is not None:
        cropped_dict['seg'] = F.grid_sample(input=seg,
                                            grid=affine_grid,
                                            mode='nearest',
                                            padding_mode='zeros',
                                            align_corners=False)

    return cropped_dict



