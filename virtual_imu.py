import json

import numpy as np
from pycocotools.coco import COCO

import articulate as art
import torch
import os.path as osp
import os
import pickle as pkl
import numpy as np

from config_test import config

subject_list = [1, 5, 6, 7, 8, 9, 11]


body_model = art.ParametricModel(
    official_model_file='smpl/SMPL_neutral.pkl')

vi_mask = torch.tensor([1961, 5424, 1176, 4662, 411, 3021])
ji_mask = torch.tensor([18, 19, 4, 5, 15, 0])

annot_path = osp.join( 'data', 'Human36M','annotations')

data_cam_smpl = "data/Human36M/data_cam_smpl"
from utils.preprocessing import load_img
from utils.transforms import bbox_from_json, crop

cameras = {}
import cv2
import smplx

smpl = smplx.create(config.smpl_path, 'smpl')



def get_smpl_coord(smpl_param, cam_param):
    # camera parameter
    R, t, f, c = np.array(cam_param['R'], dtype=np.float32), np.array(cam_param['t'], dtype=np.float32), np.array(
        cam_param['f'], dtype=np.float32), np.array(cam_param['c'], dtype=np.float32)
    cam_param = {'R': R, 't': t, 'focal': f, 'princpt': c}

    # smpl parameter
    pose, shape, trans = smpl_param['pose'], smpl_param['shape'], smpl_param['trans']
    pose = torch.FloatTensor(pose).view(-1, 3)  # (24,3)
    root_pose = pose[0, None, :]
    body_pose = pose[1:, :]
    shape = torch.FloatTensor(shape).view(1, -1)  # SMPL shape parameter
    trans = torch.FloatTensor(trans).view(1, -1)  # translation vector

    # apply camera extrinsic (rotation)
    # merge root pose and camera rotation
    root_pose = root_pose.numpy()
    root_pose, _ = cv2.Rodrigues(root_pose)
    root_pose, _ = cv2.Rodrigues(np.dot(R, root_pose))
    root_pose = torch.from_numpy(root_pose).view(1, 3)

    # change to mean shape if beta is too far from it
    shape[(shape.abs() > 3).any(dim=1)] = 0.

    # get mesh and joint coordinates
    with torch.no_grad():
        output = smpl(betas=shape, body_pose=body_pose.view(1, -1), global_orient=root_pose, transl=trans)
    mesh_cam = output.vertices[0].numpy()
    joint_cam = output.joints[0].numpy()

    # apply camera exrinsic (translation)
    # compenstate rotation (translation from origin to root joint was not cancled)
    root_cam = joint_cam[0, None, :]
    joint_cam = joint_cam - root_cam + np.dot(R, root_cam.transpose(1, 0)).transpose(1,
                                                                                     0) + t / 1000  # camera-centered coordinate system
    mesh_cam = mesh_cam - root_cam + np.dot(R, root_cam.transpose(1, 0)).transpose(1,
                                                                                   0) + t / 1000  # camera-centered coordinate system
    pose[0, None, :] = root_pose
    return joint_cam, torch.from_numpy(mesh_cam), pose, shape


def process_amass():
    for subject in subject_list:
        for ac in range(2, 17):
            for sb in range(1, 3):
                for cam_idx in range(1, 5):
                    pkl_file = f'data_cam_smpl/h36m_s{subject}_ac{ac}_sb{sb}_ca{cam_idx}.pkl'
                    with open(pkl_file, 'rb') as f:
                        datalist = pkl.load(f, encoding='latin1')
                    poses = []
                    shapes = []
                    pose_aas = []
                    verts = []
                    features = []
                    for data in datalist:
                        smpl_param = data['smpl_param']
                        cam_param = data['cam_param']
                        joint_cam, mesh_cam, pose, shape = get_smpl_coord(smpl_param, cam_param)
                        pose_aas.append(pose)
                        p = art.math.axis_angle_to_rotation_matrix(pose).view(24, 3, 3)
                    
                        poses.append(p)
                        shapes.append(shape)
                        verts.append(mesh_cam)

                    poses = torch.stack(poses)
                    pose_aas = torch.stack(pose_aas)
                    shapes = torch.stack(shapes)
                    verts = torch.stack(verts)
           

                    grot, joint = body_model.forward_kinematics(poses, shapes, calc_mesh=False)

                    orientation, acceleration = get_ori_accel(grot, verts[:, vi_mask], frame_rate=50)
                    for i in range(1, len(datalist) - 1):
                        data = datalist[i]
                        data['theta'] = pose_aas[i].numpy()
                        data['imu_acc'] = acceleration[i - 1].numpy()
                        data['imu_ori'] = orientation[i - 1].numpy()
                        data['shape'] = shapes[i].numpy()
                        data['rotation'] = poses[i].numpy()
                        data['feature'] = features[i].numpy()
                        datalist[i] = data

                    pkl_file_imu = f'features/h36m_s{subject}_ac{ac}_sb{sb}_ca{cam_idx}.pkl'
                    with open(pkl_file_imu, 'wb') as f:
                        pkl.dump(datalist, f)


def get_ori_accel(A_global_list: object, vertex: object, frame_rate: object) -> object:
    acceleration = []
    orientation = A_global_list[:, ji_mask]
    time_interval = 1.0 / frame_rate
    total_number = len(A_global_list)
    print(total_number)

    for idx in range(1, total_number - 1):
        vertex_0 = vertex[idx - 1]  # 6*3
        vertex_1 = vertex[idx]
        vertex_2 = vertex[idx + 1]
        accel_tmp = (vertex_2 + vertex_0 - 2 * vertex_1) / (time_interval * time_interval)
        acceleration.append(accel_tmp)
    return orientation[1:-1], torch.stack(acceleration)


if __name__ == '__main__':
    process_amass()
