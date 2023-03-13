import torch.utils.data
import torchvision.transforms as transforms
import os.path as osp
import pickle as pkl
import re
import numpy as np
from utils.transforms import crop
from utils.preprocessing import load_img
from utils.geometry import batch_rodrigues
from utils.transforms import normalize_and_concat_72
import config
import os

from models.smpl import TotalCapture_TO_J14, SMPL

OPEN2LSP = [2, 1, 0, 3, 4, 5]


def get_fitting_error(gt_joint, smpl_joint):
    print(np.sqrt(np.sum((gt_joint - smpl_joint) ** 2, 1)))
    error = np.sqrt(np.sum((gt_joint - smpl_joint) ** 2, 1)).mean()
    return error


class TotalCaptureDataset(torch.utils.data.Dataset):
    def __init__(self, is_train, img_res=224, window_size=16):
        super(TotalCaptureDataset, self).__init__()
        self.is_train = is_train
        self.img_res = img_res
        self.img_path = os.path.join(config.BASE_DATA_DIR, 'totalCapture')
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.window_size = window_size

        self.smpl_layer = SMPL(
            config.SMPL_MODEL_DIR,
            batch_size=64,
            create_transl=False,
        ).cpu()
        self.data_train, self.data_val, self.data_test = self.load_data()
        if self.is_train == 'train':
            self.data_all = self.data_train
        elif self.is_train == 'val':
            self.data_all = self.data_val
        else:
            self.data_all = self.data_test

        print(len(self.data_all))

    def load_data(self):
        data_list_train = []
        data_list_val = []
        data_list_test = []
        data_path1 = osp.join(self.img_path, 'totalcapture_train_s123.pkl')
        data_path2 = osp.join(self.img_path, 'totalcapture_test_s12345.pkl')
        data_path3 = osp.join(self.img_path, 'totalcapture_val_s45.pkl')

        if self.is_train == 'train':
            with open(data_path1, 'rb') as f:
                data = pkl.load(f, encoding='latin1')
                for i in range(len(data) - self.window_size):
                    if data[i]['img_idx'] % 5 == 0:
                        data_list_train.append(data[i:i + self.window_size])

            with open(data_path2, 'rb') as f:
                data = pkl.load(f, encoding='latin1')
                for i in range(len(data) - self.window_size):
                    if data[i]['subject'] == 4:
                        if data[i]['img_idx'] % 5 == 0:
                            data_list_train.append(data[i:i + self.window_size])
        if self.is_train == 'val':
            with open(data_path2, 'rb') as f:
                data = pkl.load(f, encoding='latin1')
                for i in range(len(data) - self.window_size):
                    if data[i]['subject'] == 4:
                        continue
                    if data[i]['img_idx'] % 32 == 0:
                        data_list_val.append(data[i:i + self.window_size])
            with open(data_path3, 'rb') as f:
                data = pkl.load(f, encoding='latin1')
                for i in range(len(data) - self.window_size):
                    if data[i]['img_idx'] % 32 == 0:
                        data_list_val.append(data[i:i + self.window_size])

        if self.is_train == 'test':
            #
            # with open(data_path2, 'rb') as f:
            #     data = pkl.load(f, encoding='latin1')
            #     for i in range(len(data) - self.window_size):
            #         # if data[i]['subject'] == 4:
            #         #     continue
            #         # if data[i]['subject'] == 1:
            #         #     # if data[i]['action'] == 5:
            #         #         if data[i]['img_idx'] % 64 == 0:
            #         #             data_list_test.append(data[i:i + self.window_size])
            # # with open(data_path3, 'rb') as f:
            # #     data = pkl.load(f, encoding='latin1')
            # #     for i in range(len(data) - self.window_size):
            # #
            # #         if data[i]['img_idx'] % 64 == 0:
            # #             data_list_test.append(data[i:i + self.window_size])
            with open(data_path2, 'rb') as f:
                data = pkl.load(f, encoding='latin1')
                for i in range(len(data) - self.window_size):
                    if data[i]['subject'] == 4:
                        continue
                    if data[i]['img_idx'] % 64 == 0:
                        data_list_test.append(data[i:i + self.window_size])
            with open(data_path3, 'rb') as f:
                data = pkl.load(f, encoding='latin1')
                for i in range(len(data) - self.window_size):
                    if data[i]['img_idx'] % 64 == 0:
                        data_list_test.append(data[i:i + self.window_size])

        return data_list_train, data_list_val, data_list_test

    def __len__(self):
        return len(self.data_all)

    def __getitem__(self, index):

        data_list = self.data_all[index]
        # mid_frame = int(self.window_size / 2)
        data = data_list[-1]
        img_path, center, scale = data['img_path'], data['center'], data['scale']
        img_paths = img_path.split("/")
        img_path = os.path.join(self.img_path, 'images', img_paths[-2], img_paths[-1])

        img = load_img(img_path)
        scale = max(scale[0], scale[1])
        img_test, width, center_x, center_y = crop(img, center, scale, (self.img_res, self.img_res))

        if self.transform:
            img = self.transform(img_test)

        pose = batch_rodrigues(torch.FloatTensor(data['smpl_pose']).view(-1, 3)).reshape(-1, 24, 3, 3)
        imu_root = torch.FloatTensor(data['imu_ori'][-1])

        """imu_data"""
        imu_rotations = []
        imu_accs = []
        leaf_keypoints = []
        relative_keypoints = []
        for i in range(len(data_list)):
            imu_rotation = torch.FloatTensor(data_list[i]['imu_ori'])
            imu_acc = torch.FloatTensor(data_list[i]['imu_acc'])
            leaf_keypoint = torch.FloatTensor(data_list[i]['left_keypoints'])
            relative_keypoint = torch.FloatTensor(data_list[i]['relative_keypoints'])
            imu_rotations.append(imu_rotation)
            imu_accs.append(imu_acc)
            leaf_keypoints.append(leaf_keypoint)
            relative_keypoints.append(relative_keypoint)

        imu_rotations = torch.stack(imu_rotations)
        imu_accs = torch.stack(imu_accs)
        leaf_keypoints = torch.stack(leaf_keypoints)
        relative_keypoints = torch.stack(relative_keypoints)
        imu_data = normalize_and_concat_72(imu_accs, imu_rotations)
        # vertices, th_jtr = self.smpl_layer_data(torch.from_numpy(data['smpl_pose']).float().view(1, 72))
        # print(",,,,,,,,,,,",th_jtr*1000)
        # R_world = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
        # joint_gt = np.dot(R_world, data['joints_gt'].hmr(1, 0))
        # joint_gt = joint_gt.T
        # joint_gt = joint_gt - joint_gt[:1]
        # print(joint_gt[total_12])
        # print( th_jtr[0][smpl_12].numpy()*1000.)
        #
        # error = get_fitting_error(joint_gt[total_12], th_jtr[0][smpl_12].numpy()*1000.)
        # print(error)
        smpl_shape = torch.zeros(10).float()
        smpl_output = self.smpl_layer(
            betas=smpl_shape.unsqueeze(0),
            body_pose=pose[:, 1:],
            global_orient=pose[:, 0].unsqueeze(1),
            pose2rot=False
        )
        pred_vertices = smpl_output.vertices[0]
        pred_joints = smpl_output.joints[0]

        inputs = {'img': img, 'imu_data': imu_data, 'leaf_keypoints': leaf_keypoints.view(self.window_size, -1)}
        targets = {
            'theta': torch.FloatTensor(data['smpl_pose']),
            'smpl_pose': pose[0],
            'smpl_shape': smpl_shape,
            'kp_3d': pred_joints[25:39],  # 49*3
            'verts': pred_vertices,
            'leaf_keypoints': leaf_keypoints,
            'relative_keypoints': relative_keypoints

        }

        jvel_init = torch.zeros(24 * 3)
        metor = {'root': imu_root, 'PIP_acc': imu_accs, 'PIP_rot': imu_rotations, 'init_leaf': leaf_keypoints[0],
                 "init_all": jvel_init, "image": img_test}
        info = {'img_path': img_path, 'box': torch.tensor([112., 112., 224], dtype=torch.float32),
                'verts': pred_vertices, 'gt_shape': smpl_shape, "bbox": torch.tensor([center_x, center_y, width]),
                'subject': data['subject']
                }
        return inputs, targets, metor, info
