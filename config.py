import os

# ------------------------ Paths ------------------------
# Additional files
SMPL_MODEL_DIR = 'additional/smpl'
SMPL_FACES_PATH = 'additional/smpl_faces.npy'
SMPL_MEAN_PARAMS_PATH = 'additional/neutral_smpl_mean_params_6dpose.npz'
J_REGRESSOR_EXTRA_PATH = 'additional/J_regressor_extra.npy'
COCOPLUS_REGRESSOR_PATH = 'additional/cocoplus_regressor.npy'
H36M_REGRESSOR_PATH = 'additional/J_regressor_h36m.npy'
VERTEX_TEXTURE_PATH = 'additional/vertex_texture.npy'
CUBE_PARTS_PATH = 'additional/cube_parts.npy'
DP_UV_PROCESSED_FILE = 'additional/UV_Processed.mat'
HRNET_PATH = 'additional/hrnet_results_centred.npy'

POINTREND_CONFIG = "models/third_party_models/PointRend/configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco.yaml"
DENSEPOSE_CONFIG = "models/third_party_models/DensePose/configs/densepose_rcnn_R_101_FPN_s1x.yaml"
KEYPOINTS_CONFIG = "COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"
RESNET18_PRETRAINED = "./checkpoints/resnet_pretrained.pth"
HRNET_CONFIG = "./checkpoints/pose_hrnet_w48_384x288.pth"
LITE_HRNET_CONFIG = './models/third_party_models/litehrnet/configs/litehrnet_30_coco_384x288.py'


EDGE_NMS = True
EDGE_THRESHOLD = 0.0
EDGE_GAUSSIAN_STD = 1.0
EDGE_GAUSSIAN_SIZE = 5

# Dataset Paths
DATASET_NPZ_PATH = './datasets'

TEST_SSP3D = './datasets/ssp_3d'
TEST_3DPW = './datasets/3dpw'
TEST_3DOH50K = './datasets/3doh50k'

# DATASET NPZ FILES

DATASET_FILES = [{'h36m-p1': os.path.join(DATASET_NPZ_PATH, 'h36m_valid_protocol1.npz'),
                  'h36m': os.path.join(DATASET_NPZ_PATH, 'h36m_mosh_valid_p2.npz'),
                  'mpi-inf-3dhp': os.path.join(DATASET_NPZ_PATH, 'mpi_inf_3dhp_test.npz'),
                  '3dpw': os.path.join(DATASET_NPZ_PATH, '3dpw_validation.npz'),
                  # '3dpw_test': os.path.join(DATASET_NPZ_PATH, '3dpw_test.npz'),
                  'up3d':os.path.join(DATASET_NPZ_PATH, 'up3d_val.npz'),
                  # 'up3d_test':os.path.join(DATASET_NPZ_PATH, 'up3d_test.npz'),
                  'coco_whole_body': os.path.join(DATASET_NPZ_PATH, 'coco_wholebody_val.npz'),
                  'synthetic': os.path.join(DATASET_NPZ_PATH, 'up3d_3dpw_val.npz')
                 },
                 {'h36m': os.path.join(DATASET_NPZ_PATH, 'h36m_mosh_train.npz'),
                  '3dpw': os.path.join(DATASET_NPZ_PATH, '3dpw_train.npz'),
                  'up3d':os.path.join(DATASET_NPZ_PATH, 'up3d_train.npz'),
                  'coco_whole_body': os.path.join(DATASET_NPZ_PATH, 'coco_wholebody_train.npz'),
                  'mpi-inf-3dhp': os.path.join(DATASET_NPZ_PATH, 'mpi_inf_3dhp_train.npz'),
                  'synthetic': os.path.join(DATASET_NPZ_PATH, 'amass_up3d_3dpw_train.npz')
                  }
                ]


TEXTURE_PATHS = [{'texture':os.path.join(DATASET_NPZ_PATH, 'smpl_val_textures.npz')},
{'texture':os.path.join(DATASET_NPZ_PATH, 'smpl_train_textures.npz')}]

BACKGROUNDS = [{'backgrounds':os.path.join(DATASET_NPZ_PATH, 'train_files/lsun_backgrounds/val')},
{'backgrounds':os.path.join(DATASET_NPZ_PATH, 'train_files/lsun_backgrounds/train')}]

# DATASET ROOT FILES

H36M_ROOT = '/data/angelisg/datasets/h36m_data'
PW3D_ROOT = '/data/angelisg/datasets/Mesh_Reconstruction/3dpw'
UP3D_ROOT = '/data/angelisg/datasets/Mesh_Reconstruction/up-3d'
MPI_INF_3DHP_ROOT = '/data/angelisg/datasets/Mesh_Reconstruction/mpi_inf_3dhp'
COCO_ROOT = '/data/angelisg/datasets/Mesh_Reconstruction/coco'

DATASET_FOLDERS = {'h36m': H36M_ROOT,
                   'h36m-p1': H36M_ROOT,
                   'h36m-p2': H36M_ROOT,
                   'mpi-inf-3dhp': MPI_INF_3DHP_ROOT,
                   'coco_whole_body': COCO_ROOT,
                   '3dpw': PW3D_ROOT,
                   '3dpw_val': PW3D_ROOT,
                   'up3d': UP3D_ROOT,
                   'up3d_val': UP3D_ROOT,
                   'up3d_test': UP3D_ROOT,
                }


# ------------------------ Constants ------------------------
AUGMENT_RGB = {
  'LIGHT_LOC_RANGE': [0.05, 3.0],
  'LIGHT_AMBIENT_RANGE': [0.4, 0.8],
  'LIGHT_DIFFUSE_RANGE': [0.4, 0.8],
  'LIGHT_SPECULAR_RANGE': [0.0, 0.5],
  'OCCLUDE_BOTTOM_PROB': 0.02,
  'OCCLUDE_TOP_PROB': 0.005,
  'OCCLUDE_VERTICAL_PROB': 0.05,
  'PIXEL_CHANNEL_NOISE': 0.2}



FOCAL_LENGTH = 5000.
REGRESSOR_IMG_WH = 256
HEATMAP_GAUSSIAN_STD = 4.0

# Mean and standard deviation for normalizing input image
IMG_NORM_MEAN = [0.485, 0.456, 0.406]
IMG_NORM_STD = [0.229, 0.224, 0.225]

# CAM AUGMENT PARAMETERS
XY_STD = 0.05
DELTA_Z_RANGE = [-5, 5]


CROP_INPUT = True
BBOX_AUGMENT_PARAMS = True
MEAN_SCALE_FACTOR = 1.2
DELTA_SCALE_RANGE = [-0.2, 0.2]
DELTA_CENTRE_RANGE = [-5, 5]


# ----------------- Proxy Representation Augmentation Parameters -----------------
REMOVE_APPENDAGES = True
DEVIATE_JOINTS2D = True
DEVIATE_VERTS2D = True
OCCLUDE_SEG = True
REMOVE_JOINTS = True

REMOVE_APPENDAGES_CLASSES = [1, 2, 3, 4, 5, 6]
REMOVE_APPENDAGES_PROBABILITIES = [0.4, 0.2, 0.5, 0.2, 0.15, 0.15]
DELTA_J2D_DEV_RANGE = [-8, 8]
DELTA_J2D_HIP_DEV_RANGE = [-9, 9]
DELTA_VERTS2D_DEV_RANGE = [-0.02, 0.02]
OCCLUDE_PROBABILITY = 0.7
OCCLUDE_BOX_DIM = 64
REMOVE_JOINTS_INDICES = [7, 8, 9, 10, 13, 14, 15, 16]
REMOVE_JOINTS_PROB = 0.4
EXTREME_CROP_PROB = 0.4

# IMAGE BASED OCCLUSIONS
USE_SYNTHETIC_OCCLUSION = False
OCC_AUG_DATASET = 'pascal'
COCO_OCCLUDERS_FILE = ''
PASCAL_OCCLUDERS_FILE = './additional/pascal_occluders.pkl'
# ------------------------ Joint label conventions ------------------------
# The SMPL model (im smpl_official.py) returns a large superset of joints.
# Different subsets are used during training - e.g. H36M 3D joints convention and COCO 2D joints convention.
# You may wish to use different subsets in accordance with your training data/inference needs.

# The joints superset is broken down into: 45 SMPL joints (24 standard + additional fingers/toes/face),
# 9 extra joints, 19 cocoplus joints and 17 H36M joints.
# The 45 SMPL joints are converted to COCO joints with the map below.
# (Not really sure how coco and cocoplus are related.)

# Indices to get 17 COCO joints and 17 H36M joints from joints superset.
ALL_JOINTS_TO_COCO_MAP = [24, 26, 25, 28, 27, 16, 17, 18, 19, 20, 21, 1, 2, 4, 5, 7, 8]
ALL_JOINTS_TO_H36M_MAP = list(range(73, 90))

# Indices to get the 14 LSP joints from the 17 H36M joints
H36M_TO_J17 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10, 0, 7, 9]
H36M_TO_J14 = H36M_TO_J17[:14]

BBOX_SCALE_FACTOR = 1.2
BBOX_THRESHOLD = 0.95

TWENTYFOUR_PART_SEG_TO_COCO_JOINTS_MAP = {19: 7,
                                          21: 7,
                                          20: 8,
                                          22: 8,
                                          4: 9,
                                          3: 10,
                                          12: 13,
                                          14: 13,
                                          11: 14,
                                          13: 14,
                                          5: 15,
                                          6: 16}

# ----------------- Proxy Representation Augmentation Parameters -----------------
REMOVE_APPENDAGES = True
DEVIATE_JOINTS2D = True
DEVIATE_VERTS2D = True
OCCLUDE_SEG = True


OCCLUDE_BOX_PROB = 0.1
OCCLUDE_BOTTOM_PROB = 0.2
OCCLUDE_TOP_PROB = 0.005
OCCLUDE_VERTICAL_PROB = 0.05
EXTREME_CROP_PROB = 0.1
JOINTS_TO_SWAP = [[5, 6], [11, 12]]  # COCO joint labels
JOINTS_SWAP_PROB = 0.1
PIXEL_CHANNEL_NOISE = 0.2

REMOVE_PARTS_CLASSES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]  # DensePose part classes
REMOVE_PARTS_PROBS = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.1, 0.1,
                                                            0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05]
REMOVE_APPENDAGE_JOINTS_PROB = 0.5
REMOVE_JOINTS_INDICES = [7, 8, 9, 10, 13, 14, 15, 16]  # COCO joint labels
REMOVE_JOINTS_PROB = 0.1


REMOVE_APPENDAGE_JOINTS_PROB = 0.5

SMPL_JOINTS_FLIP_PERM = [0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13, 15, 17, 16, 19, 18, 21, 20, 23, 22]
SMPL_POSE_FLIP_PERM = []
for i in SMPL_JOINTS_FLIP_PERM:
    SMPL_POSE_FLIP_PERM.append(3*i)
    SMPL_POSE_FLIP_PERM.append(3*i+1)
    SMPL_POSE_FLIP_PERM.append(3*i+2)

# Permutation indices for the 24 ground truth joints
J24_FLIP_PERM = [5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 12, 13, 14, 15, 16, 17, 18, 19, 21, 20, 23, 22]
# Permutation indices for the full set of 49 joints
J49_FLIP_PERM = [0, 1, 5, 6, 7, 2, 3, 4, 8, 12, 13, 14, 9, 10, 11, 16, 15, 18, 17, 22, 23, 24, 19, 20, 21]\
              + [25+i for i in J24_FLIP_PERM]



