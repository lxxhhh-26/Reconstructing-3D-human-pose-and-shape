HUMAN36M_DATA_DIR = '/data/3D_HPE_projects/imu_connect_images/data'
BASE_DATA_DIR = '/data/3D_HPE_projects/imu_connect_images/data'
SMPL_MODEL_DIR = '/data/3D_HPE_projects/imu_connect_images/data/smpl'

SMPL_MEAN_PARAMS = '/data/3D_HPE_projects/imu_connect_images/data/smpl_mean_params.npz'

OURS_DATA_DIR = '/data/project_HPE_3D/process_ours_data/data'

smpl_names = [
    'Left_Hip', 'Right_Hip', 'Waist', 'Left_Knee', 'Right_Knee',
    'Upper_Waist', 'Left_Ankle', 'Right_Ankle', 'Chest',
    'Left_Toe', 'Right_Toe', 'Base_Neck', 'Left_Shoulder',
    'Right_Shoulder', 'Upper_Neck', 'Left_Arm', 'Right_Arm',
    'Left_Elbow', 'Right_Elbow', 'Left_Wrist', 'Right_Wrist',
    'Left_Finger', 'Right_Finger'
]

openpose_to_smpl = [12, 9, 13, 10, 14, 11, 5, 2, 6, 3, 7, 4]
smpl_to_conf = [1, 2, 4, 5, 7, 8, 16, 17, 18, 19, 20, 21]


class joint_set:
    leaf = [7, 8, 12, 20, 21]
    full = list(range(1, 24))
    reduced = [1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19]
    ignored = [0, 7, 8, 10, 11, 20, 21, 22, 23]

    lower_body = [0, 1, 2, 4, 5, 7, 8, 10, 11]
    lower_body_parent = [None, 0, 0, 1, 2, 3, 4, 5, 6]

    n_leaf = len(leaf)
    n_full = len(full)
    n_reduced = len(reduced)
    n_ignored = len(ignored)
