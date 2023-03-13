import os
from torch.utils.tensorboard import SummaryWriter
import sys
import config
from dataset.TotalCapture import TotalCaptureDataset
import torch
import utils.log as log
import utils.utils as utils
from torch.utils.data import DataLoader, random_split
import time
import torch.backends.cudnn as cudnn
from keypoints_loss import compute_errors
from opt import Options
from last_attebtion.connect_total_last import Net_Connect
from loss_totalCapture_last import Loss_split_2d_3d

# writer
writer = SummaryWriter('./logs')
# 一机多卡设置
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
poses = []
poses_gt = []


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
    device_ids = [0, 1]  # 选中其中两块
    model = torch.nn.DataParallel(model, device_ids=device_ids)  # 并行使用两块
    model = model.cuda()
    # model.apply(weight_init)
    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))
    criterion = Loss_split_2d_3d().cuda()
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
            dataset=TotalCaptureDataset(is_train='test'),
            batch_size=1,
            shuffle=False,
            num_workers=1,
            pin_memory=True)
        _, err_test, err_imu = test(test_loader, model, criterion, save_mesh=False)

        print(">>>>>> TEST results:")
        print("{:.4f}".format(err_test), end='\t')
        print("{:.4f}".format(err_imu), end='\t')

        torch.save(poses, 'out.pt')
        torch.save(poses_gt, 'test.pt')

        sys.exit()

    # data loading
    print(">>> loading data")
    # dataset = TotalCaptureDataset(is_train='train')
    # train_len = int(0.7 * len(dataset))
    # val_len = len(dataset) - train_len
    #
    # train_dataset, val_dataset = random_split(dataset, lengths=[train_len, val_len],
    #                                            generator=torch.Generator().manual_seed(0))
    # train_loader = DataLoader(
    #     dataset=train_dataset,
    #     batch_size=64,
    #     shuffle=True,
    #     num_workers=16,
    #     pin_memory=True)
    # val_loader = DataLoader(
    #     dataset=val_dataset,
    #     batch_size=64,
    #     shuffle=True,
    #     num_workers=16,
    #     pin_memory=True)

    train_loader = DataLoader(
        dataset=TotalCaptureDataset(is_train='train'),
        batch_size=64,
        shuffle=True,
        num_workers=16,
        pin_memory=True)
    val_loader = DataLoader(
        dataset=TotalCaptureDataset(is_train='val'),
        batch_size=64,
        shuffle=True,
        num_workers=16,
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

        if epoch % 20 == 0:
            log.save_ckpt({'epoch': epoch + 1,
                           'lr': lr_now,
                           'err': err_best,
                           'state_dict': model.state_dict(),
                           'optimizer': optimizer.state_dict()},
                          ckpt_path=opt.ckpt,
                          epoch=epoch,
                          is_best=False)
    logger.close()


def train(train_loader, model, criterion, optimizer,
          lr_init=None, lr_now=None, glob_step=None, lr_decay=None, gamma=None,
          max_norm=True, epoch=0):
    losses = utils.AverageMeter()
    model.train()
    start = time.time()
    batch_time = 0
    error = 0
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


from models.smpl import SMPL

smpl_layer = SMPL(config.SMPL_MODEL_DIR,
                  batch_size=1,
                  create_transl=False)


def test(test_loader, model, criterion, epoch=0, save_mesh=False):
    losses = utils.AverageMeter()
    model.eval()
    start = time.time()
    batch_time = 0
    error = 0
    error_imu = 0
    i = 0
    for i, (inps, tars, metor, _) in enumerate(test_loader):
        targets = {key: tars[key].cuda() for key in tars}
        inputs = {key: inps[key].cuda() for key in inps}
        metor = {key: metor[key].cuda() for key in metor}

        outputs = model(inputs, imu_root=metor['root'])

        # calculate loss
        loss = criterion(outputs, targets)
        losses.update(loss.item(), inputs['img'].size(0))
        # calculate erruracy
        pred, loss_3d_pos = compute_errors(outputs['kp_3d'].detach().cpu().numpy(),
                                           targets['kp_3d'].detach().cpu().numpy())
        print(pred)

        poses_gt.append(targets['smpl_pose'].cpu())
        poses.append(outputs['smpl_pose'].detach().cpu())
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
    print(i)
    e1 = (error / (i + 1)) * 1000
    e2 = (error_imu / (i + 1)) * 1000
    print(">>> error: {} <<<".format(e1))
    print(">>> error: {} <<<".format(e2))

    return losses.avg, e1, e2


if __name__ == "__main__":
    option = Options().parse()
    main(option)
