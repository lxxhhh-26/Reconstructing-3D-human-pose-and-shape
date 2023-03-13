import torch.utils.data
import torchvision.transforms as transforms
import os.path as osp
import pickle as pkl
import re
import numpy as np
from utils.transforms import crop
from utils.preprocessing import load_img
from utils.geometry import batch_rodrigues, rotation_matrix_to_angle_axis
from utils.transforms import normalize_and_concat_72, normalize_screen_coordinates, normalize_and_concat, bbox_from_json
import config
import os
from smplpytorch.pytorch.smpl_layer import SMPL_Layer
from config import joint_set
from models.smpl import H36M_TO_J14, SMPL
import cv2

cam_param = {"R": [[0.9228353966173104, -0.37440015452287667, 0.09055029013436408],
                   [-0.01498208436370467, -0.269786590656035, -0.9628035794752281],
                   [0.38490306298896904, 0.8871525910436372, -0.25457791009093983]],
             "t": [25.76853374383657, 431.05581759025813, 4461.872981411145],
             "f": [1145.51133842318, 1144.77392807652],
             "c": [514.968197319863, 501.882018537695]}


class H36MDataset(torch.utils.data.Dataset):
    def __init__(self, is_train, img_res=224, window_size=16):
        super(H36MDataset, self).__init__()
        self.is_train = is_train
        self.img_res = img_res
        self.window_size = window_size
        self.img_path = os.path.join(config.BASE_DATA_DIR, 'h36m')
        # self.data_path_imu = os.path.join(self.img_path, "imu")
        self.data_path_imu = os.path.join(config.BASE_DATA_DIR, 'h36m', "imu_3d")
        self.data_path_pkl = os.path.join(self.img_path, "h36m")

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.smpl_layer = SMPL(
            config.SMPL_MODEL_DIR,
            batch_size=64,
            create_transl=False,
        ).cpu()
        self.h36m_joint_regressor = np.load(
            config.BASER_FOUDER + 'Paper-Project/connect_features/dataset/J_regressor_h36m_correct.npy')
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

        if self.is_train == 'train':
            subjects = [1, 5, 6, 7, 8]
            for s in subjects:
                for ac in range(2, 17):
                    for sb in range(1, 3):
                        pkl_path = f'h36m_s{s}_action{ac}_subaction{sb}_imu_3d.pkl'
                        pkl_path = os.path.join(self.data_path_imu, pkl_path)
                        with open(pkl_path, 'rb') as f:
                            data_imu = pkl.load(f, encoding='latin1')
                        data_path = f'h36m_s{s}_action{ac}_subaction{sb}.pkl'
                        pkl_data = os.path.join(self.data_path_pkl, data_path)
                        with open(pkl_data, 'rb') as f:
                            data_data = pkl.load(f, encoding='latin1')
                        data_data = data_data[1:-1]
                        for i in range(len(data_imu) - self.window_size):
                            if i % 5 != 0:
                                continue
                            data = {'img': data_data[i + self.window_size - 1],
                                    'imu': data_imu[i: i + self.window_size]}
                            data_list_train.append(data)

        if self.is_train == 'val':
            subjects = [9, 11]
            for s in subjects:
                for ac in range(2, 17):
                    for sb in range(1, 3):
                        pkl_path = f'h36m_s{s}_action{ac}_subaction{sb}_imu_3d.pkl'
                        pkl_path = os.path.join(self.data_path_imu, pkl_path)
                        with open(pkl_path, 'rb') as f:
                            data_imu = pkl.load(f, encoding='latin1')
                        data_path = f'h36m_s{s}_action{ac}_subaction{sb}.pkl'
                        pkl_data = os.path.join(self.data_path_pkl, data_path)
                        with open(pkl_data, 'rb') as f:
                            data_data = pkl.load(f, encoding='latin1')
                        data_data = data_data[1:-1]
                        for i in range(len(data_imu) - self.window_size):
                            if i % 64 != 0:
                                continue
                            data = {'img': data_data[i + self.window_size - 1],
                                    'imu': data_imu[i: i + self.window_size]}
                            data_list_val.append(data)

        if self.is_train == 'test':
            # s_11_act_16_subact_02_ca_04
            s = 11
            a = 16
            sb = 2
            pkl_path = f'h36m_s{s}_action{a}_subaction{sb}_imu_3d.pkl'
            pkl_path = os.path.join(self.data_path_imu, pkl_path)
            with open(pkl_path, 'rb') as f:
                data_imu = pkl.load(f, encoding='latin1')
            data_path = f'h36m_s{s}_action{a}_subaction{sb}.pkl'
            pkl_data = os.path.join(self.data_path_pkl, data_path)
            with open(pkl_data, 'rb') as f:
                data_data = pkl.load(f, encoding='latin1')
            data_data = data_data[1:-1]
            for i in range(len(data_imu) - self.window_size):
                if i % 2 != 0:
                    continue
                data = {'img': data_data[i + self.window_size - 1],
                        'imu': data_imu[i: i + self.window_size]}
                data_list_test.append(data)

            # subjects = [9, 11]
            # for s in subjects:
            #     for ac in range(2, 17):
            #         for sb in range(1, 3):
            #             pkl_path = f'h36m_s{s}_action{ac}_subaction{sb}_imu_3d.pkl'
            #             pkl_path = os.path.join(self.data_path_imu, pkl_path)
            #             with open(pkl_path, 'rb') as f:
            #                 data_imu = pkl.load(f, encoding='latin1')
            #             data_path = f'h36m_s{s}_action{ac}_subaction{sb}.pkl'
            #             pkl_data = os.path.join(self.data_path_pkl, data_path)
            #             with open(pkl_data, 'rb') as f:
            #                 data_data = pkl.load(f, encoding='latin1')
            #             data_data = data_data[1:-1]
            #
            #             for i in range(len(data_imu) - self.window_size):
            #                 if i % 64 != 0:
            #                     continue
            #                 data = {'img': data_data[i + self.window_size - 1],
            #                         'imu': data_imu[i: i + self.window_size]}
            #                 data_list_test.append(data)

        return data_list_train, data_list_val, data_list_test

    def get_smpl_coord(self, smpl_param, cam_param):

        pose, shape, trans = smpl_param['pose'], smpl_param['shape'], smpl_param['trans']
        smpl_pose = torch.FloatTensor(pose).view(-1, 3)
        smpl_shape = torch.FloatTensor(shape).view(1, -1)  # smpl parameters (pose: 72 dimension, shape: 10 dimension)

        R, t = np.array(cam_param['R'], dtype=np.float32).reshape(3, 3), np.array(cam_param['t'],
                                                                                  dtype=np.float32).reshape(
            3)  # camera rotation and translation

        # merge root pose and camera rotation
        root_pose = smpl_pose[0, :].numpy()
        root_pose, _ = cv2.Rodrigues(root_pose)
        root_pose, _ = cv2.Rodrigues(np.dot(R, root_pose))
        smpl_pose[0] = torch.from_numpy(root_pose).view(3)
        pose = batch_rodrigues(smpl_pose.view(-1, 3)).reshape(-1, 24, 3, 3)

        # get mesh and joint coordinates
        # smpl_poses = smpl_pose.view(24, 3).numpy()
        # rotation = []
        # for p in smpl_poses:
        #     rotation.append(cv2.Rodrigues(p)[0])
        #
        # rotation = np.array(rotation)
        # rotation = torch.tensor(rotation, dtype=torch.float32).view(-1, 24, 3, 3)

        # smplout = self.smpl(global_orient=rotation[:, 0].unsqueeze(1), body_pose=rotation[:, 1:], betas=smpl_shape,
        #                     pose2rot=False)
        #
        smplout = self.smpl_layer(
            betas=smpl_shape,
            body_pose=pose[:, 1:],
            global_orient=pose[:, 0].unsqueeze(1),
            pose2rot=False
        )

        # smplout = self.smpl(global_orient=rotation[:, 0].unsqueeze(1), body_pose=rotation[:, 1:], betas=smpl_shape,
        #                     transl=trans, pose2rot=False)

        smpl_mesh_coord = smplout.vertices
        smpl_joint_coord = smplout.joints
        # incorporate face keypoints
        smpl_mesh_coord = smpl_mesh_coord.detach().numpy().astype(np.float32).reshape(-1, 3)
        smpl_joint_coord = smpl_joint_coord.detach().numpy().astype(np.float32).reshape(-1, 3)

        # compenstate rotation (translation from origin to root joint was not cancled)
        # smpl_trans = np.array(trans, dtype=np.float32).reshape(
        #     3)  # translation vector from smpl coordinate to h36m world coordinate
        # smpl_trans = np.dot(R, smpl_trans[:, None]).reshape(1, 3) + t.reshape(1, 3) / 1000
        # root_joint_coord = smpl_joint_coord[0].reshape(1, 3)
        # smpl_trans = smpl_trans - root_joint_coord + np.dot(R, root_joint_coord.transpose(1, 0)).transpose(1, 0)
        #
        # smpl_mesh_coord = smpl_mesh_coord + smpl_trans
        # smpl_joint_coord = smpl_joint_coord + smpl_trans

        # change to mean shape if beta is too far from it
        smpl_shape[(smpl_shape.abs() > 3).any(dim=1)] = 0.

        # meter -> milimeter
        smpl_mesh_coord *= 1000
        smpl_joint_coord *= 1000
        return smpl_mesh_coord

    def get_fitting_error(self, h36m_joint, smpl_mesh):

        h36m_joint = h36m_joint - h36m_joint[0, None, :]  # root-relative
        h36m_from_smpl = np.dot(self.h36m_joint_regressor, smpl_mesh)
        h36m_from_smpl = h36m_from_smpl - np.mean(h36m_from_smpl, 0)[None, :] + np.mean(h36m_joint, 0)[None,
                                                                                :]  # translation alignment

        # h36m_from_smpl = h36m_from_smpl - h36m_from_smpl[self.h36m_root_joint_idx, None, :]

        error = np.sqrt(np.sum((h36m_joint - h36m_from_smpl) ** 2, 1)).mean()
        return error

    def __len__(self):
        return len(self.data_all)

    def __getitem__(self, index):

        data_list = self.data_all[index]
        data = data_list['img']
        data_imus = data_list['imu']
        center, scale = bbox_from_json(data['bbox'])

        img_path = data['img_path']
        img_path = os.path.join(self.img_path, 'occlusion', img_path)
        # img_path = os.path.join(self.img_path, 'images', img_path)
        # cv2.imshow("on", cv2.imread(img_path))

        # cv2.waitKey(0)
        img = load_img(img_path)
        img_test, width, w, h = crop(img, center, scale, (self.img_res, self.img_res))

        """get_error"""
        smpl_param = data['smpl_param']
        # smpl_mesh_cam = self.get_smpl_coord(smpl_param, cam_param)
        # error = self.get_fitting_error(data['joint_cam'], smpl_mesh_cam)
        # print(error)
        # print(error)
        if self.transform:
            img = self.transform(img_test)

        imu_rotations = []
        imu_accs = []
        left_Jtrs = []
        relative_Jtrs = []

        for data_imu in data_imus:
            imu_rotation = torch.FloatTensor(data_imu['imu_ori'])
            imu_acc = torch.FloatTensor(data_imu['imu_acc'])
            left_Jtr = torch.FloatTensor(data_imu['left_keypoints'])
            relative_Jtr = torch.FloatTensor(data_imu['relative_keypoints'])

            left_Jtrs.append(left_Jtr)
            relative_Jtrs.append(relative_Jtr)
            imu_rotations.append(imu_rotation)
            imu_accs.append(imu_acc)

        imu_accs = torch.stack(imu_accs)
        imu_rotations = torch.stack(imu_rotations)
        left_Jtrs = torch.stack(left_Jtrs)
        relative_Jtrs = torch.stack(relative_Jtrs)

        imu_data = normalize_and_concat_72(imu_accs, imu_rotations)
        pose_aa = torch.FloatTensor(data_imus[-1]['pose']).view(-1, 72)
        pose = batch_rodrigues(pose_aa.view(-1, 3)).reshape(-1, 24, 3, 3)
        smpl_shape = torch.FloatTensor(smpl_param['shape'])
        # pred_shape_zero = torch.zeros(10).to(smpl_shape.device).expand_as(smpl_shape)

        smpl_output = self.smpl_layer(
            betas=smpl_shape.unsqueeze(0),
            body_pose=pose[:, 1:],
            global_orient=pose[:, 0].unsqueeze(1),
            pose2rot=False
        )
        pred_joints = smpl_output.joints[0][25:39]
        pred_vertices = smpl_output.vertices

        inputs = {'img': img, 'imu_data': imu_data}
        targets = {
            'theta': pose_aa.view(-1, 3),
            'smpl_shape': smpl_shape,
            'smpl_pose': pose[0],
            'kp_3d': pred_joints,  # 14*3
            'verts': pred_vertices[0],
            'left_keypoints': left_Jtrs,
            'relative_keypoints': relative_Jtrs,
        }
        metor = {'root': imu_rotations[-1, -1], 'cam_joint': data['joint_cam'],
                 'smpl_image': torch.FloatTensor(smpl_param['pose']),
                 'smpl_trans': torch.FloatTensor(smpl_param['trans']), }

        info = {'img_path': img_path, 'smpl_param': smpl_param, 'cam_param': data['cam_param'], 'center': center,
                'scale': scale}

        return inputs, targets, metor, info
