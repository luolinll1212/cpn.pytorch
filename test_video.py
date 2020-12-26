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
from src.log import AverageMeter
from src import utils

from config import config as cfg

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def draw_point(model, image):
    img_h, img_w, _ = image.shape
    # 图片缩放
    scale = cfg.img_size / max(img_h, img_w)
    img_h2, img_w2 = int(img_h * scale), int(img_w * scale)
    img = cv.resize(image, (img_w2, img_h2))
    # 格式转换
    img = np.transpose(img, (2, 0, 1)).astype(np.float32)  # H*W*C --> C*H*W
    img[[0, 2]] = img[[2, 0]]  # BGR --> RGB
    img = img / 255.0  # 归一化
    # 填充0
    pad_imgs = np.zeros(shape=(3, cfg.img_size, cfg.img_size))
    pad_imgs[:, :img.shape[1], :img.shape[2]] = img
    img = torch.Tensor([pad_imgs]) # numpy -> torch

    # cpu -> gpu
    img = img.to(device, dtype=torch.float32)
    out, p2 = model(img)  # model 输出

    # keypoints list
    kps = []
    for h in out[0]:
        pos = h.view(-1).argmax().item()
        x = (pos % h.size(1)) * cfg.img_hp_rate
        y = (pos // h.size(1)) * cfg.img_hp_rate
        kps.append([x, y])

    # (224*224)kps -> (1920*1080) kps
    kps_image = []
    for i in kps:
        x, y  = i[0], i[1]
        x = int(x / cfg.img_size * 1920)
        y = int(y / cfg.img_size * 1920)
        kps_image.append([x,y])

    # 绘制点
    image = utils.draw_cv_kps(image, kps_image)  # image对象
    # cv.imwrite("video/1.jpg", image, [int(cv.IMWRITE_JPEG_QUALITY), 95])
    # exit()

    return image


def cut_video(model, src, dst):
    # 读取视频
    video = cv.VideoCapture(src)
    fps = video.get(cv.CAP_PROP_FPS)  # 帧率，每秒多少帧
    minute = int(fps) * 60  # 每分钟的帧数

    # 保存视频
    fourcc = cv.VideoWriter_fourcc(*'XVID')  # 保存视频的编码
    out = cv.VideoWriter(dst, fourcc, fps, (1920, 1080))

    # 读取视频
    i = 0
    while (video.isOpened()):
        ret, frame = video.read()  # 读取图片

        if frame is not None:
            # 视频处理
            image = draw_point(model, frame)
            # print(image.shape)
            out.write(image)
        else:
            pass

        # 显示分钟数
        if i % minute == 0:
            now_minute = i // minute
            print("pass {} minutes".format(now_minute))

        if not ret:  # 条件终止
            break

        i += 1  # 遍历视频

    video.release()
    out.release()


if __name__ == '__main__':
    # 视频流
    src = r"video/all_flow.avi"
    dst = r"video/01.avi"

    # 模型
    checkoutpth = r"checkout/keypoint02.2020-11-23.epoch.100.pt"
    model = cpn_resnet50(cfg.num_kps)
    model.load_state_dict(torch.load(checkoutpth))
    model.eval()
    model.to(device)

    # 读取视频
    cut_video(model, src, dst)
