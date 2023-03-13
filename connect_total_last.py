import os

import torch.nn as nn
import torch
import os.path as osp
from last_attebtion.HMR_total_last import get_pretrained_hmr, Regressor
from config import BASE_DATA_DIR
from last_attebtion.Transpose_total_last import get_imu_model
from models.net_utils import FC


class Net_Connect(nn.Module):
    def __init__(
            self,
    ):
        super(Net_Connect, self).__init__()

        self.hmr_model = get_pretrained_hmr()
        self.imu_model = get_imu_model()
        self.fc_full_pose = FC(87, 1024, dropout_r=0.2)
        self.regressor = Regressor()

    def forward(self, input, imu_root=None):
        xf_img = self.hmr_model.feature_extractor(input['img'])

        full_joint_position, leaf_keypoints = self.imu_model(input['imu_data'])

        full_joint_position = self.fc_full_pose(full_joint_position)

        output = self.regressor(xf_img, full_joint_position, imu_root=imu_root)
        output['leaf_keypoints'] = leaf_keypoints

        return output
