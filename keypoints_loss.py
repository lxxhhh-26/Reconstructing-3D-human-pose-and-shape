# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import numpy as np


def align_by_pelvis(joints, get_pelvis=False):
    """
    Assumes joints is 14 x 3 in LSP order.
    Then hips are: [2, 3]
    Takes mid point of these points, then subtracts it.
    """
    left_id = 2
    right_id = 3

    pelvis = (joints[left_id, :] + joints[right_id, :]) / 2.
    if get_pelvis:
        return joints - np.expand_dims(pelvis, axis=0), pelvis
    else:
        return joints - np.expand_dims(pelvis, axis=0)


def compute_errors(gt3ds, preds):
    """
    Gets MPJPE after pelvis alignment + MPJPE after Procrustes.
    Evaluates on the 14 common joints.
    Inputs:
      - gt3ds: N x 14 x 3
      - preds: N x 14 x 3

    'Right Ankle', 'Right Knee', 'Right Hip',
    'Left Hip', 'Left Knee', 'Left Ankle',
    'Right Wrist', 'Right Elbow', 'Right Shoulder',
    'Left Shoulder', 'Left Elbow', 'Left Wrist',
    'Neck (LSP)', 'Top of Head (LSP)',
    """
    errors = []
    target = [13]
    for i, (gt3d, pred) in enumerate(zip(gt3ds, preds)):
        gt3d = gt3d.reshape(-1, 3)
        pred = pred.reshape(-1, 3)
        # Root align.
        gt3d = align_by_pelvis(gt3d)[target]
        pred = align_by_pelvis(pred)[target]

        joint_error = np.sqrt(np.sum((gt3d - pred) ** 2, axis=1))

        # smpl_names = [
        #     'Left_Hip', 'Right_Hip', 'Waist', 'Left_Knee', 'Right_Knee',
        #     'Upper_Waist', 'Left_Ankle', 'Right_Ankle', 'Chest',
        #     'Left_Toe', 'Right_Toe', 'Base_Neck', 'Left_Shoulder',
        #     'Right_Shoulder', 'Upper_Neck', 'Left_Arm', 'Right_Arm',
        #     'Left_Elbow', 'Right_Elbow', 'Left_Wrist', 'Right_Wrist',
        #     'Left_Finger', 'Right_Finger'
        # ]
        # print(smpl_names)
        errors.append(np.mean(joint_error))
    errors_mean = np.stack(errors).mean()
    return errors, errors_mean


def compute_errors_24(gt3ds, preds):
    """
    Gets MPJPE after pelvis alignment + MPJPE after Procrustes.
    Evaluates on the 14 common joints.
    Inputs:
      - gt3ds: N x 14 x 3
      - preds: N x 14 x 3
    """
    errors = []
    for i, (gt3d, pred) in enumerate(zip(gt3ds, preds)):
        gt3d = gt3d.reshape(-1, 3)
        pred = pred.reshape(-1, 3)
        # Root align.

        joint_error = np.sqrt(np.sum((gt3d - pred) ** 2, axis=1))
        # smpl_names = [
        #     'Left_Hip', 'Right_Hip', 'Waist', 'Left_Knee', 'Right_Knee',
        #     'Upper_Waist', 'Left_Ankle', 'Right_Ankle', 'Chest',
        #     'Left_Toe', 'Right_Toe', 'Base_Neck', 'Left_Shoulder',
        #     'Right_Shoulder', 'Upper_Neck', 'Left_Arm', 'Right_Arm',
        #     'Left_Elbow', 'Right_Elbow', 'Left_Wrist', 'Right_Wrist',
        #     'Left_Finger', 'Right_Finger'
        # ]
        # print(smpl_names)
        errors.append(np.mean(joint_error))

    errors_mean = np.stack(errors).mean()

    return errors, errors_mean


def mpjpe(predicted, target):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape

    pred = np.linalg.norm(predicted - target, axis=len(target.shape) - 1)
    # joint_error = torch.sqrt(torch.sum((predicted - target)**2, axis=1)).mean()
    # print(joint_error)
    return np.mean(pred)


def weighted_mpjpe(predicted, target, w):
    """
    Weighted mean per-joint position error (i.e. mean Euclidean distance)
    """
    assert predicted.shape == target.shape
    assert w.shape[0] == predicted.shape[0]
    return torch.mean(w * torch.norm(predicted - target, dim=len(target.shape) - 1))


def p_mpjpe(predicted, target):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    """
    assert predicted.shape == target.shape

    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)

    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0 ** 2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0 ** 2, axis=(1, 2), keepdims=True))

    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1))  # Rotation

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY  # Scale
    t = muX - a * np.matmul(muY, R)  # Translation

    # Perform rigid transformation on the input
    predicted_aligned = a * np.matmul(predicted, R) + t

    # Return MPJPE
    return np.mean(np.linalg.norm(predicted_aligned - target, axis=len(target.shape) - 1))


def n_mpjpe(predicted, target):
    """
    Normalized MPJPE (scale only), adapted from:
    https://github.com/hrhodin/UnsupervisedGeometryAwareRepresentationLearning/blob/master/losses/poses.py
    """
    assert predicted.shape == target.shape

    norm_predicted = torch.mean(torch.sum(predicted ** 2, dim=3, keepdim=True), dim=2, keepdim=True)
    norm_target = torch.mean(torch.sum(target * predicted, dim=3, keepdim=True), dim=2, keepdim=True)
    scale = norm_target / norm_predicted
    return mpjpe(scale * predicted, target)


def mean_velocity_error(predicted, target):
    """
    Mean per-joint velocity error (i.e. mean Euclidean distance of the 1st derivative)
    """
    assert predicted.shape == target.shape

    velocity_predicted = np.diff(predicted, axis=0)
    velocity_target = np.diff(target, axis=0)

    return np.mean(np.linalg.norm(velocity_predicted - velocity_target, axis=len(target.shape) - 1))
