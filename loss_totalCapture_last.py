import torch.nn as nn
from models.smpl import TotalCapture_TO_J14, H36M_TO_J14


class Loss_split_2d_3d(nn.Module):
    def __init__(self):
        super(Loss_split_2d_3d, self).__init__()
        self.shape_loss_weight = 0.01
        self.keypoint_loss_weight = 10
        self.pose_loss_weight = 2
        self.pose_loss_betas = 0
        self.keypoint_all_relative = 0.01

        # Per-vertex loss on the shape
        self.criterion_shape = nn.L1Loss()
        # Keypoint (2D and 3D) loss
        # No reduction because confidence weighting needs to be applied
        self.criterion_keypoints = nn.MSELoss()
        # Loss for SMPL parameter regression
        self.criterion_regr = nn.MSELoss()
        # Loss for relative_keypoints
        self.criterion_relative = nn.MSELoss()

    def keypoint_3d_loss_relative(self, pred_keypoints_3d, gt_keypoints_3d, conf=None):
        """Compute 3D keypoint loss for the examples that 3D keypoint annotations are available.
        The loss is weighted by the confidence.
        """

        loss = self.criterion_relative(pred_keypoints_3d.float(),
                                       gt_keypoints_3d.float())
        return loss

    def keypoint_3d_loss(self, pred_keypoints_3d, gt_keypoints_3d, conf=None):
        """Compute 3D keypoint loss for the examples that 3D keypoint annotations are available.
        The loss is weighted by the confidence.
        """

        gt_pelvis = (gt_keypoints_3d[:, 2, :] + gt_keypoints_3d[:, 3, :]) / 2
        gt_keypoints_3d = gt_keypoints_3d - gt_pelvis[:, None, :]

        pred_pelvis = (pred_keypoints_3d[:, 2, :] + pred_keypoints_3d[:, 3, :]) / 2
        pred_keypoints_3d = pred_keypoints_3d - pred_pelvis[:, None, :]
        loss = self.criterion_keypoints(pred_keypoints_3d.float(),
                                        gt_keypoints_3d.float())
        return loss

    def shape_loss(self, pred_vertices, gt_vertices):
        """Compute per-vertex loss on the shape for the examples that SMPL annotations are available."""
        return self.criterion_shape(pred_vertices, gt_vertices)

    def smpl_losses(self, pred_rotmat, gt_pose):
        loss_regr_pose = self.criterion_regr(pred_rotmat, gt_pose)

        return loss_regr_pose

    def forward(self, output, target):
        # Compute loss on SMPL parameters
        loss_regr_pose = self.smpl_losses(output['smpl_pose'],
                                          target['smpl_pose'])
        # # Compute 3D keypoint loss
        loss_keypoints_gt = self.keypoint_3d_loss(output['kp_3d'], target['kp_3d'])
        loss_vertices = self.shape_loss(output['verts'],
                                        target['verts'])
        loss_keypoints_leaf = self.keypoint_3d_loss_relative(output['leaf_keypoints'],
                                                             target['leaf_keypoints'])
        # loss_keypoints_relative = self.keypoint_3d_loss_relative(output['relative_keypoints'],
        #                                                          target['relative_keypoints'])

        # Compute total loss
        # The last component is a loss that forces the network to predict positive depth values
        loss = self.keypoint_loss_weight * loss_keypoints_gt + \
               self.shape_loss_weight * loss_vertices + \
               self.pose_loss_weight * loss_regr_pose + \
               self.keypoint_all_relative * loss_keypoints_leaf
        loss *= 60
        losses = {'loss': loss.detach().item(),
                  'loss_keypoints': loss_keypoints_gt.detach().item(),
                  'loss_vertices': loss_vertices.detach().item(),
                  'loss_pose': loss_regr_pose.detach().item(),
                  'loss_leaf': loss_keypoints_leaf.detach().item(),
                  }
        print(losses)
        return loss
