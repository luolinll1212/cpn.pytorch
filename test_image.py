# -*- coding: utf-8 -*-
import torch
import numpy as np
import cv2 as cv

from models.network import cpn_resnet50
from src import utils

from config import config as cfg

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def show_keypoint(model, src, dst):
    # 读取图片
    image = cv.imread(src)
    # 图片转换，维度，归一化
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

    image = utils.draw_cv_kps(image, kps_image)  # image对象
    cv.imwrite(dst, image, [int(cv.IMWRITE_JPEG_QUALITY), 95])


if __name__ == '__main__':
    # 视频流
    src = r"image/1757430.jpg"
    dst = r"image/01.jpg"

    # 模型
    checkoutpth = r"checkout/keypoint02.2020-11-23.epoch.100.pt"
    model = cpn_resnet50(cfg.num_kps)
    model.load_state_dict(torch.load(checkoutpth))
    model.eval()
    model.to(device)

    # 关键点截取
    show_keypoint(model, src, dst)

