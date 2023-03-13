import torch
from torch.nn.functional import relu
from config import joint_set
import smplx


class RNN(torch.nn.Module):
    r"""
    An RNN Module including a linear input layer, an RNN, and a linear output layer.
    """

    def __init__(self, n_input, n_output, n_hidden, n_rnn_layer=2, bidirectional=True, dropout=0.2):
        super(RNN, self).__init__()
        self.rnn = torch.nn.LSTM(n_hidden, n_hidden, n_rnn_layer, bidirectional=bidirectional)
        self.linear1 = torch.nn.Linear(n_input, n_hidden)
        self.linear2 = torch.nn.Linear(n_hidden * (2 if bidirectional else 1), n_output)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, h=None, feature=False):
        self.rnn.flatten_parameters()
        x, h = self.rnn(relu(self.linear1(self.dropout(x))), h)
        if feature:
            return x
        else:
            return self.linear2(x), h


class TransPoseNet(torch.nn.Module):
    def __init__(self):
        super(TransPoseNet, self).__init__()

        n_imu = 6 * 3 + 6 * 9  # acceleration (vector3) and rotation matrix (matrix3x3) of 6 IMUs
        self.pose_s1 = RNN(n_imu, joint_set.n_leaf * 3, 256)

    def forward(self, x):
        batch, len_size, data_size = x.shape
        imu = x.permute(1, 0, 2)
        leaf_joint_position_out = self.pose_s1.forward(imu)[0]

        leaf_joint_position = leaf_joint_position_out.permute(1, 0, 2).reshape(batch, len_size, -1, 3)

        return torch.cat((leaf_joint_position_out, imu), dim=2), leaf_joint_position


import os
from config import BASE_DATA_DIR


def get_imu_model(pred_weight=os.path.join(BASE_DATA_DIR, 'weights.pt')):
    model = TransPoseNet()
    checkpoint = torch.load(pred_weight)
    model.load_state_dict(checkpoint, strict=False)

    return model
