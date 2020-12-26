# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import os, datetime
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import cv2 as cv

from config import config as cfg
from dataLoader import KeyPointDataset
from models.network import cpn_resnet50
from loss.cpn_loss import CPNLoss
from src import utils, log, miou, tool

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(model, criterion, optimizer, train_loader, epoch, cfg):
    model.train()
    global_loss_am = []
    refine_loss_am = []
    all_loss_am = []
    all_miou_am = []
    # 迭代数据
    for batch_idx, (img, hp, masks, kps) in enumerate(train_loader):
        img, hp, masks = img.to(device), hp.to(device), masks.to(device)  # cpu -> gpu

        # 网络输出
        out, p2 = model(img)
        all_loss, global_loss, refine_loss = criterion(p2, out, hp, masks)
        # 异常处理
        if torch.isnan(all_loss):
            print("Loss exploded at step {}".format(epoch))
            raise Exception("Loss exploded")

        # oprimizer
        optimizer.zero_grad()
        all_loss.backward()
        optimizer.step()

        # 输出预测值，计算miou
        pred_kps = tool.compute_model_kps(out.data.cpu()) # 预测点，三维
        rate = miou.compute_miou(pred_kps, kps[:,:,0:2].tolist())
        all_miou_am += [(np.mean(rate))]

        # 损失迭加
        global_loss_am += [(global_loss.item())]
        refine_loss_am += [(refine_loss.item())]
        all_loss_am += [(all_loss.item())]



        if batch_idx % cfg.interval_print == 0:
            print("{} peoch, {} step, all loss is {:.4f}, global loss is {:.4f}, refine loss is {:.4f}, miou is {:.4f}".format(
                epoch, batch_idx, np.mean(all_loss_am), np.mean(global_loss_am), np.mean(refine_loss_am), np.mean(all_miou_am)))
            # 更新损失
            global_loss_am = []
            refine_loss_am = []
            all_loss_am = []

    # 返回
    return np.mean(all_loss_am), np.mean(global_loss_am), np.mean(refine_loss_am), np.mean(all_miou_am)


def test(model, criterion, test_loader, cfg, epoch):
    model.eval()
    test_all_loss = 0.
    test_global_loss = 0.
    test_refine_loss = 0.
    test_miou = []
    save_path = f"{cfg.checkout}/{epoch}"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for batch_idx, (img, hp, masks, kps) in enumerate(test_loader):
        # 保存图片，注意，batch=1
        image = utils.torch2np2image(img)  # torch -> numpy, BGR

        img, hp, masks = img.to(device), hp.to(device), masks.to(device)  # cpu -> gpu

        # cpu -> gpu
        imgs = img.to(device, dtype=torch.float32)
        out, p2 = model(imgs)  # model 输出
        all_loss, global_loss, refine_loss = criterion(p2, out, hp, masks)

        # 损失迭代
        test_all_loss += all_loss.item()
        test_global_loss += global_loss.item()
        test_refine_loss += refine_loss.item()

        # 网络预测 keypoints list
        pred_kps = []
        for h in out[0]:
            pos = h.view(-1).argmax().item()
            x = (pos % h.size(1)) * cfg.img_hp_rate
            y = (pos // h.size(1)) * cfg.img_hp_rate
            pred_kps.append([x, y])
        # 计算miou
        rate = miou.compute_miou([pred_kps], kps[:,:,0:2].tolist())
        test_miou += [(rate)]

        # 绘制点
        image = utils.draw_kps(image, pred_kps, cfg)  # image对象
        image.save(f"{save_path}/{batch_idx}.jpg")

    # 损失
    print("all loss is {:.4f}, global loss is {:.4f}, refine loss is {:.4f}, miou is {:.4f}".format(
        test_all_loss / len(test_loader), test_global_loss / len(test_loader), test_refine_loss / len(test_loader), np.mean(test_miou)))
    # 返回
    return test_all_loss / len(test_loader), test_global_loss / len(test_loader), test_refine_loss / len(test_loader), np.mean(test_miou)


if __name__ == '__main__':
    # 输出
    if not os.path.exists(cfg.checkout):
        os.mkdir(cfg.checkout)
    # 日志
    if not os.path.exists(cfg.logs):
        os.mkdir(cfg.logs)

    # 训练集
    train_set = KeyPointDataset(cfg, label_txt=cfg.train_list, mode="train")
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, num_workers=cfg.num_workers,
                              shuffle=True, drop_last=False, collate_fn=train_set.collate_fn)

    # 测试集，保存图片，batch_size=1
    test_set = KeyPointDataset(cfg, label_txt=cfg.test_list)
    test_loader = DataLoader(test_set, batch_size=1, num_workers=cfg.num_workers,
                             shuffle=True, drop_last=False, collate_fn=test_set.collate_fn)

    # 开始的批次
    start_epoch = cfg.start_epoch

    # 网络
    model = cpn_resnet50(cfg.num_kps, pretrained=cfg.pretrained)
    # 预训练
    if cfg.checkoutpth != "":
        print("load checkoutpth model:{}".format(cfg.checkoutpth))
        model.load_state_dict(torch.load(cfg.checkoutpth))
    model.to(device)
    # 多卡
    if torch.cuda.device_count() > 1:
        device_ids = range(torch.cuda.device_count())
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        model.to(device)


    # 损失函数
    criterion = CPNLoss(num_kps=cfg.num_kps)

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    # 日志
    logger = log.Logger(cfg.logs)

    for epoch in range(start_epoch, cfg.num_epochs + 1):
        # 学习率衰减
        if epoch in [cfg.num_epochs * 0.25, cfg.num_epochs * 0.5, cfg.num_epochs * 0.75]:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
        # 训练
        train_all_loss, train_global_loss, train_refine_loss, train_miou = train(model, criterion, optimizer, train_loader, epoch,
                                                                     cfg)
        logger.log(step=epoch, content={"train_all_loss": train_all_loss,
                                        "train_global_loss": train_global_loss,
                                        "train_refine_loss": train_refine_loss,
                                        "miou":train_miou})  # 日志
        # 测试
        test_all_loss, test_global_loss, test_refine_loss, test_miou = test(model, criterion, test_loader, cfg, epoch)
        logger.log(step=epoch, content={"test_all_loss": test_all_loss,
                                        "test_global_loss": test_global_loss,
                                        "test_refine_loss": test_refine_loss,
                                        "miou": test_miou})  # 日志
        # 保存，is_best
        if test_miou > cfg.is_best:
            cfg.is_best = test_miou
            pth_name = "keypoint02.best.{0}.epoch.{1}.pt".format(datetime.datetime.now().strftime("%F"), epoch)
            torch.save(model.state_dict(), f"{cfg.checkout}/{pth_name}")

        # 间隔保存
        if epoch % cfg.interval_save == 0:
            pth_name = "keypoint02.{0}.epoch.{1}.pt".format(datetime.datetime.now().strftime("%F"), epoch)
            torch.save(model.state_dict(), f"{cfg.checkout}/{pth_name}")
