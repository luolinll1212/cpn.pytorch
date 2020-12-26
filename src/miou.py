# -*- coding: utf-8 -*-
import numpy as np
from config import config as cfg
import cv2

def miou(pred, target, nclass=1):
    mini = 1

    # 计算公共区域
    intersection = pred * (pred == target)

    # 直方图
    area_inter, _ = np.histogram(intersection, bins=2, range=(mini, nclass))
    area_pred, _ = np.histogram(pred, bins=2, range=(mini, nclass))
    area_target, _ = np.histogram(target, bins=2, range=(mini, nclass))
    area_union = area_pred + area_target - area_inter

    # 交集已经小于并集
    assert (area_inter <= area_union).all(), "Intersection area should be smaller than Union area"

    rate = round(max(area_inter) / max(area_union), 4)
    return rate


def compute_miou(kp_pred, kp_target):
    rates = []
    for kp_p, kp_t in zip(kp_pred, kp_target):
        #
        mask_pred = np.zeros(shape=(cfg.img_size, cfg.img_size))
        mask_target = np.zeros(shape=(cfg.img_size, cfg.img_size))
        # 四个点
        point_pred = np.array([kp_p], np.int32)
        point_target = np.array([kp_t], np.int32)

        # mask
        res_mask_pred = cv2.fillPoly(mask_pred, point_pred, 1)
        res_mask_target = cv2.fillPoly(mask_target, point_target, 1)

        # 计算miou
        rate = miou(res_mask_pred, res_mask_target)
        rates += [(rate)]

    # 返回miou列表
    return rates




if __name__ == '__main__':
    # nclass = 1
    # # target
    # target = np.zeros(shape=(200, 200))
    # target[0:100, 0:100] = 1
    #
    # # pred
    # pred = np.zeros(shape=(200, 200))
    # pred[10:110, 10:110] = 1
    #
    # # 计算miou
    # rate = miou(pred, target, nclass)
    # print(rate)

    # 计算miou
    kp_pred = [[[0, 0], [100, 0], [100, 100], [0, 100]]]
    kp_target = [[[10, 10], [110, 10], [110, 110], [10, 110]]]
    rates = compute_miou(kp_pred, kp_target)
    print(rates)
