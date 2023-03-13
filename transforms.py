import numpy
import torch
import numpy as np
import numpy as np
import scipy.misc
import cv2
from PIL import Image


def cam2pixel(cam_coord, f, c):
    x = cam_coord[:, 0] / cam_coord[:, 2] * f[0] + c[0]
    y = cam_coord[:, 1] / cam_coord[:, 2] * f[1] + c[1]
    z = cam_coord[:, 2]
    return np.stack((x, y, z), 1)


def pixel2cam(pixel_coord, f, c):
    x = (pixel_coord[:, 0] - c[0]) / f[0] * pixel_coord[:, 2]
    y = (pixel_coord[:, 1] - c[1]) / f[1] * pixel_coord[:, 2]
    z = pixel_coord[:, 2]
    return np.stack((x, y, z), 1)


def world2cam(world_coord, R, t):
    cam_coord = np.dot(R, world_coord.transpose(1, 0)).transpose(1, 0) + t.reshape(1, 3)
    return cam_coord


def cam2world(cam_coord, R, t):
    world_coord = np.dot(np.linalg.inv(R), (cam_coord - t.reshape(1, 3)).transpose(1, 0)).transpose(1, 0)
    return world_coord


def rigid_transform_3D(A, B):
    n, dim = A.shape
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    H = np.dot(np.transpose(A - centroid_A), B - centroid_B) / n
    U, s, V = np.linalg.svd(H)
    R = np.dot(np.transpose(V), np.transpose(U))
    if np.linalg.det(R) < 0:
        s[-1] = -s[-1]
        V[2] = -V[2]
        R = np.dot(np.transpose(V), np.transpose(U))

    varP = np.var(A, axis=0).sum()
    c = 1 / varP * np.sum(s)

    t = -np.dot(c * R, np.transpose(centroid_A)) + np.transpose(centroid_B)
    return c, R, t


def rigid_align(A, B):
    c, R, t = rigid_transform_3D(A, B)
    A2 = np.transpose(np.dot(c * R, np.transpose(A))) + t
    return A2


def transform_joint_to_other_db(src_joint, src_name, dst_name):
    src_joint_num = len(src_name)
    dst_joint_num = len(dst_name)

    new_joint = np.zeros(((dst_joint_num,) + src_joint.shape[1:]), dtype=np.float32)
    for src_idx in range(len(src_name)):
        name = src_name[src_idx]
        if name in dst_name:
            dst_idx = dst_name.index(name)
            new_joint[dst_idx] = src_joint[src_idx]

    return new_joint


"""后加的"""


def bbox_from_json(bbox):
    """Get center and scale of bounding box from bounding box annotations.
    The expected format is [top_left(x), top_left(y), width, height].
    """
    x, y, width, height = bbox[0], bbox[1], bbox[2], bbox[3]
    bbox = [x, y, width, height]
    bbox = np.array(bbox).astype(np.float32)

    ul_corner = bbox[:2]
    center = ul_corner + 0.5 * bbox[2:]
    width = max(bbox[2], bbox[3])
    scale = width / 200.0
    # make sure the bounding box is rectangular
    return center, scale


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
        rot = -rot  # To match direction of rotation from cropping
        rot_mat = np.zeros((3, 3))
        rot_rad = rot * np.pi / 180
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0, :2] = [cs, -sn]
        rot_mat[1, :2] = [sn, cs]
        rot_mat[2, 2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0, 2] = -res[1] / 2
        t_mat[1, 2] = -res[0] / 2
        t_inv = t_mat.copy()
        t_inv[:2, 2] *= -1
        t = np.dot(t_inv, np.dot(rot_mat, np.dot(t_mat, t)))
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
    ul = np.array(transform([1, 1], center, scale, res, invert=1)) - 1
    # Bottom right point
    br = np.array(transform([res[0] + 1,
                             res[1] + 1], center, scale, res, invert=1)) - 1

    # Padding so that when rotated proper amount of context is included
    pad = int(np.linalg.norm(br - ul) / 2 - float(br[1] - ul[1]) / 2)
    if not rot == 0:
        ul -= pad
        br += pad

    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    if len(img.shape) > 2:
        new_shape += [img.shape[2]]
    new_img = np.zeros(new_shape)

    # Range to fill new array
    new_x = max(0, -ul[0]), min(br[0], len(img[0])) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], len(img)) - ul[1]
    # Range to sample from original image
    old_x = max(0, ul[0]), min(len(img[0]), br[0])
    old_y = max(0, ul[1]), min(len(img), br[1])
    new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1],
                                                    old_x[0]:old_x[1]]
    if not rot == 0:
        # Remove padding
        new_img = scipy.misc.imrotate(new_img, rot)

        new_img = new_img[pad:-pad, pad:-pad]
    # new_img = scipy.misc.imresize(new_img, res)
    # new_img = resize(new_img, res)
    new_img = numpy.array(Image.fromarray((np.uint8(new_img))).resize(res))
    width = max((old_y[1] - old_y[0]), (old_x[1] - old_x[0]))
    return new_img, width, old_x[0], old_y[0]


"""计算3d keypoints"""


def compute_h36m_3d_keypoints(regression_path, smpl_mesh):
    """"
    :param
    regression_path : 回归矩阵《24,6980》
    smpl_mesh:smpl_vertices，单位：毫米 《B，6980，3》
    :return
    """
    h36m_joint_regressor = np.load(regression_path).astype(np.float32)
    h36m_joint_regressor = torch.from_numpy(h36m_joint_regressor)
    h36m_joint_regressor = h36m_joint_regressor.to(smpl_mesh.device)

    h36m_from_smpls = []
    for mesh in smpl_mesh:
        # smpl_mesh[i] = smpl_mesh[i].squeeze(0)

        h36m_from_smpl = torch.mm(h36m_joint_regressor, mesh * 1000)
        h36m_from_smpl = h36m_from_smpl.cpu().detach().numpy()
        h36m_from_smpls.append(h36m_from_smpl)

    h36m_from_smpls = np.array(h36m_from_smpls).reshape(smpl_mesh.shape[0], -1, 3)
    h36m_from_smpls = torch.tensor(h36m_from_smpls, dtype=torch.float32)
    return h36m_from_smpls


"""smpl_mesh_cam"""


def get_smpl_coord(smpl_model, batch_pose, batch_shape, trans, regression_path):
    pose = batch_pose.view(-1, 24, 3, 3)
    shape = batch_shape.view(-1, 10)
    trans = trans.to(pose.device)
    trans = trans.view(pose.shape[0], -1, 3)

    smplout = smpl_model(global_orient=pose[:, 0].unsqueeze(1), body_pose=pose[:, 1:], betas=shape)
    smpl_mesh_coords = smplout.vertices
    # smpl_mesh_coords = smpl_mesh_coords + trans
    h36m_from_smpls = compute_h36m_3d_keypoints(regression_path, smpl_mesh_coords, device=pose.device)

    return h36m_from_smpls


"""tets_2"""


def get_smpl_coord_2(smpl_model, batch_pose, shape, regression_path):
    pose = batch_pose.view(-1, 24, 3, 3)
    shape = shape.view(-1, 10)
    smplout = smpl_model(global_orient=pose[:, 0].unsqueeze(1), betas=shape, body_pose=pose[:, 1:])
    smpl_mesh_coords = smplout.vertices

    h36m_from_smpls = compute_h36m_3d_keypoints(regression_path, smpl_mesh_coords)

    return h36m_from_smpls


"""save_obj"""


def save_obj(v, f, file_name='output.obj'):
    obj_file = open(file_name, 'w')
    for i in range(len(v)):
        obj_file.write('v ' + str(v[i][0]) + ' ' + str(v[i][1]) + ' ' + str(v[i][2]) + '\n')
    for i in range(len(f)):
        obj_file.write('f ' + str(f[i][0] + 1) + '/' + str(f[i][0] + 1) + ' ' + str(f[i][1] + 1) + '/' + str(
            f[i][1] + 1) + ' ' + str(f[i][2] + 1) + '/' + str(f[i][2] + 1) + '\n')
    obj_file.close()


"""angle_to_rotation"""


def angle_to_tation(smpl_pose, R=None):
    """
    :param smpl_pose: tensor[24,3]
    :param R: camera_param
    :return:
    """
    if R is not None:
        # merge root pose and camera rotation
        root_pose = smpl_pose[0, :].numpy()
        root_pose, _ = cv2.Rodrigues(root_pose)
        root_pose, _ = cv2.Rodrigues(np.dot(R, root_pose))
        smpl_pose[0] = torch.from_numpy(root_pose).view(3)
        smpl_pose = smpl_pose.view(1, -1)

    # get mesh and joint coordinates
    smpl_poses = smpl_pose.view(24, 3).numpy()
    rotation = []
    for p in smpl_poses:
        rotation.append(cv2.Rodrigues(p)[0])

    rotation = np.array(rotation)
    rotation = torch.tensor(rotation, dtype=torch.float32).view(-1, 24, 3, 3)
    return rotation


"""normalize 2d keypoints"""


def normalize_screen_coordinates(X, w, h):
    assert X.shape[-1] == 2
    # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio
    # Normalize

    X[:, 0] = X[:, 0] / float(w) - 0.5
    X[:, 1] = X[:, 1] / float(h) - 0.5
    return X * 2


def normalize_and_concat_72(glb_acc, glb_ori):
    glb_acc = glb_acc.view(-1, 6, 3)
    glb_ori = glb_ori.view(-1, 6, 3, 3)
    acc = torch.cat((glb_acc[:, :5] - glb_acc[:, 5:], glb_acc[:, 5:]), dim=1).bmm(glb_ori[:, -1]) / 30
    ori = torch.cat((glb_ori[:, 5:].transpose(2, 3).matmul(glb_ori[:, :5]), glb_ori[:, 5:]), dim=1)
    data = torch.cat((acc.flatten(1), ori.flatten(1)), dim=1)

    return data


def normalize_and_concat(glb_ori):
    glb_ori = glb_ori.view(-1, 6, 3, 3)
    ori = torch.cat((glb_ori[:, 5:].transpose(2, 3).matmul(glb_ori[:, :5]), glb_ori[:, 5:]), dim=1)
    return ori
