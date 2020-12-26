# -*- coding: utf-8 -*-
import numpy as np
from PIL import Image, ImageDraw
import cv2 as cv


# 训练测试，batch=1，torch -> numpy -> image
def torch2np2image(img):
    npImg = img.squeeze().numpy().transpose(1, 2, 0)
    npImg = (npImg * 255).astype(np.uint8)
    image = Image.fromarray(npImg) # Image
    return image


def get_kps(kpts):
    kpts = kpts.squeeze().numpy()
    kps = kpts[:, 0:2]
    return kps


def draw_kps(img, kps, cfg):
    img_h, img_w, _ = np.array(img).shape
    # get image scale rate
    scale_x, scale_y = img_w / cfg.img_size, img_h / cfg.img_size
    draw = ImageDraw.Draw(img)
    # draw points
    for kp in kps:
        x = int(scale_x * kp[0])
        y = int(scale_y * kp[1])
        # cv.circle(img, (int(x), int(y)), 4, (0, 255, 0), -1)
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(255, 0, 0, 255))
    return img

def draw_cv_kps(img, kps_image):
    for i in kps_image:
        cv.circle(img, (i[0], i[1]), 10, (0, 0, 255), -1)
    return img
