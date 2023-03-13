import os

import cv2
import smplx
from torch.utils.tensorboard import SummaryWriter
import sys
import config
from dataset.human36m import H36MDataset
import torch
import utils.log as log
import utils.utils as utils
from torch.utils.data import DataLoader
import time
import torch.backends.cudnn as cudnn
from keypoints_loss import compute_errors, p_mpjpe
from models.smpl import SMPL, H36M_TO_J14
from opt import Options
from human36m_weight.connect_h36 import Net_Connect
from loss_h36m_weight import Loss_H36m
import numpy as np

# writer
writer = SummaryWriter('./logs')
# 一机多卡设置

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
poses = []
poses_gt = []
from smplpytorch.pytorch.smpl_layer import SMPL_Layer
from utils.geometry import batch_rodrigues

smpl_layer_data = SMPL_Layer(model_root=config.SMPL_MODEL_DIR).cpu()

h36m_joint_regressor = np.load('human36m_weight/J_regressor_h36m.npy')
from draw_h36M import render_mesh

#
# cam_param = {"R": [[0.9228353966173104, -0.37440015452287667, 0.09055029013436408],
#                    [-0.01498208436370467, -0.269786590656035, -0.9628035794752281],
#                    [0.38490306298896904, 0.8871525910436372, -0.25457791009093983]],
#              "t": [25.76853374383657, 431.05581759025813, 4461.872981411145],
#              "f": [1145.51133842318, 1144.77392807652],
#              "c": [514.968197319863, 501.882018537695]}
cam_param = {'R': [[0.9154607080837831, -0.39734606500700814, 0.06362229623477154],
                   [-0.049406284684695274, -0.2678916756611978, -0.9621814117644814],
                   [0.3993628813352506, 0.877695935238897, -0.26487569589663096]],
             't': [-69.27125529438378, 422.18433660888445, 4457.893374979774],
             'f': [1145.51133842318, 1144.77392807652],
             'c': [514.968197319863, 501.882018537695]}

smpl_layer = SMPL(
    config.SMPL_MODEL_DIR,
    batch_size=64,
    create_transl=False,
).cpu()

h36m_joint_regressor = np.load(config.BASER_FOUDER+'Paper-Project/connect_features/dataset/J_regressor_h36m_correct.npy')

from utils.transforms import crop


def get_smpl_coord(pose, smpl_shape, root_pose, cam_param):
    # pose, shape, trans = smpl_param['pose'], smpl_param['shape'], smpl_param['trans']
    # smpl_pose = torch.FloatTensor(pose).view(-1, 3)
    # smpl_shape = torch.FloatTensor(shape).view(1, -1)  # smpl parameters (pose: 72 dimension, shape: 10 dimension)

    R, T = np.array(cam_param['R'], dtype=np.float32).reshape(3, 3), np.array(cam_param['t'],
                                                                              dtype=np.float32).reshape(3)  # camera rotation and translation
    root_pose = root_pose.reshape(24, 3)
    root_pose = root_pose[0]
    # merge root pose and camera rotation
    # root_pose = smpl_pose[0, :].numpy()
    root_pose, _ = cv2.Rodrigues(root_pose)
    root_pose = torch.from_numpy(np.dot(R, root_pose)).unsqueeze(0)

    # root_pose, _ = cv2.Rodrigues(np.dot(R, root_pose))
    # smpl_pose[0] = torch.from_numpy(root_pose).view(3)
    # pose = batch_rodrigues(smpl_pose.view(-1, 3)).reshape(-1, 24, 3, 3)

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
    smplout = smpl_layer(
        betas=smpl_shape,
        body_pose=pose[:, 1:],
        global_orient=root_pose.unsqueeze(1),
        pose2rot=False,
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


def get_fitting_error(h36m_joint, smpl_mesh):
    h36m_joint = h36m_joint.reshape(-1, 3)
    h36m_joint = h36m_joint - h36m_joint[0, None, :]  # root-relative
    h36m_from_smpl = np.dot(h36m_joint_regressor, smpl_mesh)
    # h36m_from_smpl = h36m_from_smpl - np.mean(h36m_from_smpl, 0)[None, :] + np.mean(h36m_joint, 0)[None,
    #                                                                         :]  # translation alignment
    h36m_from_smpl = h36m_from_smpl - h36m_from_smpl[0, None, :]  #
    p_error = p_mpjpe(torch.from_numpy(h36m_from_smpl).unsqueeze(0).numpy(),
                      torch.from_numpy(h36m_joint).unsqueeze(0).numpy())

    error = np.sqrt(np.sum((h36m_joint[H36M_TO_J14] - h36m_from_smpl[H36M_TO_J14]) ** 2, 1)).mean()

    return p_error, error


"draw========================================="


def get_smpl_coord_draw(pose, smpl_shape, root_pose, cam_param, trans, img_path):
    # pose, shape, trans = smpl_param['pose'], smpl_param['shape'], smpl_param['trans']
    # smpl_pose = torch.FloatTensor(pose).view(-1, 3)
    # smpl_shape = torch.FloatTensor(shape).view(1, -1)  # smpl parameters (pose: 72 dimension, shape: 10 dimension)
    R, t = np.array(cam_param['R'], dtype=np.float32).reshape(3, 3), np.array(cam_param['t'],
                                                                              dtype=np.float32).reshape(
        3)  # camera rotation and translation
    root_pose = root_pose.reshape(24, 3)
    root_pose = root_pose[0]
    # merge root pose and camera rotation
    # root_pose = smpl_pose[0, :].numpy()
    root_pose, _ = cv2.Rodrigues(root_pose)
    root_pose = torch.from_numpy(np.dot(R, root_pose)).unsqueeze(0)

    # root_pose, _ = cv2.Rodrigues(np.dot(R, root_pose))
    # smpl_pose[0] = torch.from_numpy(root_pose).view(3)
    # pose = batch_rodrigues(smpl_pose.view(-1, 3)).reshape(-1, 24, 3, 3)

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
    smplout = smpl_layer(
        betas=smpl_shape,
        body_pose=pose[:, 1:],
        global_orient=root_pose.unsqueeze(1),
        # transl=trans,
        pose2rot=False,
    )

    # smplout = self.smpl(global_orient=rotation[:, 0].unsqueeze(1), body_pose=rotation[:, 1:], betas=smpl_shape,
    #                     transl=trans, pose2rot=False)

    smpl_mesh_coord = smplout.vertices[0].numpy()
    smpl_joint_coord = smplout.joints[0].numpy()
    # incorporate face keypoints
    # smpl_mesh_coord = smpl_mesh_coord.detach().numpy().astype(np.float32).reshape(-1, 3)
    # smpl_joint_coord = smpl_joint_coord.detach().numpy().astype(np.float32).reshape(-1, 3)

    root_cam = smpl_joint_coord[0, None, :]
    mesh_cam = smpl_mesh_coord - root_cam + np.dot(R, root_cam.transpose(1, 0)).transpose(1,
                                                                                          0) + t / 1000  # camera-centered coordinate system
    img = cv2.imread(img_path)
    rendered_img = render_mesh(img, mesh_cam, smpl_layer.faces, cam_param)
    cv2.imwrite('smpl.jpg', rendered_img)
    print(l)
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

    # meter -> milimeter
    # smpl_mesh_coord *= 1000
    # smpl_joint_coord *= 1000
    # return smpl_mesh_coord


def demo(smpl_param, cam_param, img_path, pred_pose, pred_smpl, index, center, scale):
    smpl_layer_x = smplx.create(config.BASE_DATA_DIR, 'smpl')
    pred_smpl[(pred_smpl.abs() > 3).any(dim=1)] = 0.

    print(pred_pose.shape)
    # camera parameter
    R, t, f, c = np.array(cam_param['R'], dtype=np.float32), np.array(cam_param['t'], dtype=np.float32), np.array(
        cam_param['f'], dtype=np.float32), np.array(cam_param['c'], dtype=np.float32)
    cam_param = {'R': R, 't': t, 'focal': f, 'princpt': c}

    # smpl parameter
    pose, shape, trans = smpl_param['pose'], smpl_param['shape'], smpl_param['trans']
    pose = torch.FloatTensor(pose).view(-1, 3)  # (24,3)
    root_pose = pose[0, None, :]
    # body_pose = pred_pose[0][1:, :]
    body_pose = pose[1:, :]
    shape = torch.FloatTensor(shape).view(1, -1)  # SMPL shape parameter
    trans = torch.FloatTensor(trans).view(1, -1)  # translation vector

    # apply camera extrinsic (rotation)
    # merge root pose and camera rotation
    root_pose = root_pose.numpy()
    root_pose, _ = cv2.Rodrigues(root_pose)
    root_pose, _ = cv2.Rodrigues(np.dot(R, root_pose))
    root_pose = torch.from_numpy(root_pose).view(1, 3)

    # get mesh and joint coordinates
    with torch.no_grad():
        output = smpl_layer_x(betas=shape, body_pose=body_pose.view(1, -1), global_orient=root_pose, transl=trans)
    mesh_cam = output.vertices[0].numpy()
    joint_cam = output.joints[0].numpy()

    # apply camera exrinsic (translation)
    # compenstate rotation (translation from origin to root joint was not cancled)
    root_cam = joint_cam[0, None, :]
    joint_cam = joint_cam - root_cam + np.dot(R, root_cam.transpose(1, 0)).transpose(1,
                                                                                     0) + t / 1000  # camera-centered coordinate system
    mesh_cam = mesh_cam - root_cam + np.dot(R, root_cam.transpose(1, 0)).transpose(1,
                                                                                   0) + t / 1000  # camera-centered coordinate system

    # mesh render
    # img_path = f'images/smpl_{index}.jpg'
    img = cv2.imread(img_path)
    img_test, width, w, h = crop(img, center[0].tolist(), scale[0], (224, 224))
    # rendered_img = render_mesh(img, mesh_cam, smpl_layer.faces, cam_param)
    cv2.imwrite(f'img_ori/smpl_{index}.jpg', img_test)


def main(opt):
    start_epoch = 0
    err_best = 1000
    glob_step = 0
    lr_now = opt.lr

    # save options
    log.save_options(opt, opt.ckpt)

    # create model
    print(">>> creating model")
    model = Net_Connect()
    device_ids = [0]  # 选中其中两块
    model = torch.nn.DataParallel(model, device_ids=device_ids)  # 并行使用两块
    model = model.cuda()
    # model.apply(weight_init)
    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))
    criterion = Loss_H36m().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-5)

    # load ckpt
    if opt.load:
        print(">>> loading ckpt from '{}'".format(opt.load))
        ckpt = torch.load(opt.load)
        start_epoch = ckpt['epoch']
        err_best = ckpt['err']
        lr_now = ckpt['lr']
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        print(">>> ckpt loaded (epoch: {} | err: {})".format(start_epoch, err_best))
    if opt.resume:
        logger = log.Logger(os.path.join(opt.ckpt, 'log.txt'), resume=True)
    else:
        logger = log.Logger(os.path.join(opt.ckpt, 'log_only_imu.txt'))
        logger.set_names(['epoch', 'lr', 'loss_train', 'err_train', 'loss_test', 'err_test'])

    # test

    if opt.test:
        test_loader = DataLoader(
            dataset=H36MDataset(is_train='test'),
            batch_size=1,
            shuffle=False,
            num_workers=1,
            pin_memory=True)
        _, err_test, err_imu = test(test_loader, model, criterion, save_mesh=False)

        print(">>>>>> TEST results:")
        print("{:.4f}".format(err_test), end='\t')
        print("{:.4f}".format(err_imu), end='\t')

        # torch.save(poses, 'out.pt')
        # torch.save(poses_gt, 'test.pt')

        sys.exit()

    # data loading
    print(">>> loading data")
    # dataset = TotalCaptureDataset(is_train='train')
    # train_len = int(0.9 * len(dataset))
    # val_len = len(dataset) - train_len
    #
    # train_dataset, test_dataset = random_split(dataset, lengths=[train_len, val_len],
    #                                            generator=torch.Generator().manual_seed(0))

    train_loader = DataLoader(
        dataset=H36MDataset(is_train='train'),
        batch_size=256,
        shuffle=True,
        num_workers=4,
        pin_memory=True)
    val_loader = DataLoader(
        dataset=H36MDataset(is_train='val'),
        batch_size=164,
        shuffle=True,
        num_workers=4,
        pin_memory=True)
    print(">>> data loaded !")
    cudnn.benchmark = True
    for epoch in range(start_epoch, opt.epochs):
        print('==========================')
        print('>>> epoch: {} | lr: {:.5f}'.format(epoch + 1, lr_now))

        # per epoch
        glob_step, lr_now, loss_train, err_train = train(
            train_loader, model, criterion, optimizer,
            lr_init=opt.lr, lr_now=lr_now, glob_step=glob_step, lr_decay=opt.lr_decay, gamma=opt.lr_gamma,
            max_norm=opt.max_norm, epoch=epoch)

        loss_test, err_test, _ = test(val_loader, model, criterion, epoch=epoch)

        logger.append([epoch + 1, lr_now, loss_train, err_train, loss_test, err_test],
                      ['int', 'float', 'float', 'float', 'flaot', 'float'])

        # save ckpt
        is_best = err_test < err_best
        err_best = min(err_test, err_best)
        if is_best:
            log.save_ckpt({'epoch': epoch + 1,
                           'lr': lr_now,
                           'err': err_best,
                           'state_dict': model.state_dict(),
                           'optimizer': optimizer.state_dict()},
                          ckpt_path=opt.ckpt,
                          epoch=epoch,
                          is_best=True)

    logger.close()


def train(train_loader, model, criterion, optimizer,
          lr_init=None, lr_now=None, glob_step=None, lr_decay=None, gamma=None,
          max_norm=True, epoch=0):
    losses = utils.AverageMeter()
    model.train()
    start = time.time()
    batch_time = 0
    error = 0
    i = 0
    for i, (inps, tars, metor, _) in enumerate(train_loader):

        targets = {key: tars[key].cuda() for key in tars}
        inputs = {key: inps[key].cuda() for key in inps}
        metor = {key: metor[key].cuda() for key in metor}

        outputs = model(inputs, imu_root=metor['root'])
        # calculate loss
        optimizer.zero_grad()
        loss = criterion(outputs, targets)
        losses.update(loss.item(), inputs['img'].size(0))
        loss.backward()
        optimizer.step()

        # calculate erruracy

        pred, loss_3d_pos = compute_errors(outputs['kp_3d'].detach().cpu().numpy(),
                                           targets['kp_3d'].detach().cpu().numpy())

        error += loss_3d_pos.item()

        # update summary
        if (i + 1) % 10 == 0:
            batch_time = time.time() - start
            start = time.time()

        print('({batch}/{size}) | batch: {batchtime:.4}ms | loss: {loss:.4f}' \
              .format(batch=i + 1,
                      size=len(train_loader),
                      batchtime=batch_time * 10.0,
                      loss=losses.avg))
    # writer.add_scalar("TRAIN loss -->", losses.avg, epoch)
    e1 = (error / (i + 1)) * 1000

    return epoch, lr_now, losses.avg, e1


def test(test_loader, model, criterion, epoch=0, save_mesh=False):
    losses = utils.AverageMeter()
    model.eval()
    start = time.time()
    batch_time = 0
    error = 0
    error_imu = 0
    i = 0
    for i, (inps, tars, metor, info) in enumerate(test_loader):
        targets = {key: tars[key].cuda() for key in tars}
        inputs = {key: inps[key].cuda() for key in inps}
        metor = {key: metor[key].cuda() for key in metor}

        outputs = model(inputs, imu_root=metor['root'])

        # cam_mesh = get_smpl_coord(outputs['smpl_pose'].detach().cpu(), outputs['smpl_shape'].detach().cpu(),
        #                           metor['smpl_image'].cpu().numpy(), cam_param)
        #
        # p_error, m_error = get_fitting_error(metor['cam_joint'].cpu().numpy(), cam_mesh)
        # error_imu += p_error
        # error += m_error

        if save_mesh:
            # get_smpl_coord_draw(outputs['smpl_pose'].detach().cpu(), outputs['smpl_shape'].detach().cpu(),
            #                     metor['smpl_image'].cpu().numpy(), cam_param, metor['smpl_trans'].cpu(),
            #                     info['img_path'][0])
            demo(info['smpl_param'], info['cam_param'], info['img_path'][0], outputs['theta'].detach().cpu(),
                 outputs['smpl_shape'].detach().cpu(), i, info['center'], info['scale'])
        # calculate loss
        loss = criterion(outputs, targets)
        losses.update(loss.item(), inputs['img'].size(0))
        # calculate erruracy
        pred, loss_3d_pos = compute_errors(outputs['kp_3d'].detach().cpu().numpy(),
                                           targets['kp_3d'].detach().cpu().numpy())
        error += loss_3d_pos.item()


        if (i + 1) % 10 == 0:
            batch_time = time.time() - start
            start = time.time()

        print('({batch}/{size}) | batch: {batchtime:.4}ms| loss: {loss:.6f}' \
              .format(batch=i + 1,
                      size=len(test_loader),
                      batchtime=batch_time * 10.0,
                      loss=losses.avg))

    # writer.add_scalar("VAL loss -->", losses.avg, epoch)
    e1 = (error / (i + 1))
    e2 = (error_imu / (i + 1))
    print(">>> error: {} <<<".format(e1))
    print(">>> error: {} <<<".format(e2))

    return losses.avg, e1, e2


if __name__ == "__main__":
    option = Options().parse()
    main(option)
