# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2

from src import encode


class KeyPointDataset(Dataset):
    def __init__(self, cfg, label_txt, mode=None):
        self.cfg = cfg
        self.mode = mode
        # 归一化参数
        self.mu = np.reshape(np.asarray(self.cfg.mu), (3, 1, 1))
        self.sigma = np.reshape(np.asarray(self.cfg.sigma), (3, 1, 1))
        self.encoder = encode.KeypointEncoder()
        # 加载标签文件
        self.lines = open(label_txt, "r", encoding="utf-8").readlines()

    def __len__(self):
        return len(self.lines)

    def get_landmask(self, points):
        # 拿到所有关键点 -> 2d-array
        landmask = []
        for i in range(4):
            x = points[i * 2]
            y = points[i * 2 + 1]
            landmask += [(x, y)]
        landmask = np.array(landmask, dtype=np.float32)

        # 2d-array -> 3d-array
        kps = np.zeros((4, 3))
        kps[:, 0:2] = landmask

        return kps

    def get_data_augument(self, img, kps):
        img_h, img_w, _ = img.shape
        # 数据增强
        random_flip = np.random.randint(0, 2)
        if random_flip:
            img = cv2.flip(img, 1)  # 水平翻转
            kps[:, 0] = img_w - kps[:, 0]
        # rotation
        angle = np.random.randint(-10, 10)
        M = cv2.getRotationMatrix2D(center=(img_w / 2, img_h / 2), angle=angle, scale=1)
        img = cv2.warpAffine(img, M, dsize=(img_w, img_h), flags=cv2.INTER_CUBIC)
        kps[:, 2] = 1
        kps[:, 0:2] = np.matmul(kps, M.T)

        return img, kps

    def collate_fn(self, batch):
        imgs, kpts = zip(*batch)
        pad_imgs = torch.zeros(len(imgs), 3, self.cfg.img_size, self.cfg.img_size)
        heatmaps, vis_masks = [], []
        for i, img in enumerate(imgs):
            # If the image is smaller than `img_size`, we pad it with 0s.
            # This allows all images to have the same size.
            pad_imgs[i, :, :img.size(1), :img.size(2)] = img

            # For each image, create heatmaps and visibility masks.
            img_heatmaps, img_vis_masks = self.encoder.encode(kpts[i],
                                                              self.cfg.img_size,
                                                              self.cfg.hm_stride,
                                                              self.cfg.hm_alpha,
                                                              self.cfg.hm_sigma)

            # TODO: Can I avoid appending and do everything in torch?
            heatmaps.append(img_heatmaps)
            vis_masks.append(img_vis_masks)

        heatmaps = torch.stack(heatmaps)  # [batch_size, num_keypoints, h, w]
        vis_masks = torch.stack(vis_masks)  # [batch_size, num_keypoints]
        kpts = torch.stack(kpts)
        return pad_imgs, heatmaps, vis_masks, kpts

    def __getitem__(self, item):
        line = self.lines[item].strip().split(" ")
        img_path = line[0]
        img = cv2.imread(img_path)
        img_h, img_w, _ = img.shape

        # 关键点 -> 三位数组
        kps = self.get_landmask(line[1:])

        # 数据增强
        if self.mode == "train":
            img, kps = self.get_data_augument(img, kps)

        # 图片缩放 --> 等比例缩放, (1080x1920) -> (126,224)
        scale = self.cfg.img_size / max(img_h, img_w)
        img_h2, img_w2 = int(img_h * scale), int(img_w * scale)
        img = cv2.resize(img, (img_w2, img_h2))
        kps[:, 0:2] *= scale

        # 图片缩放 --> 等比例缩放, (1080x1920) -> (224,224)
        # scale_w, scale_h = self.cfg.img_size / img_w, self.cfg.img_size / img_h  # 得到缩放比例
        # img = cv2.resize(img, (self.cfg.img_size, self.cfg.img_size))     # 图片缩放
        # kps[:, 0], kps[:, 1] = kps[:, 0] * scale_w, kps[:, 1] * scale_h   # 点缩放

        # 格式转换
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)  # H*W*C --> C*H*W
        img[[0, 2]] = img[[2, 0]]  # BGR --> RGB
        # 归一化
        img = img / 255.0
        # img = (img - self.mu) / self.sigma

        return torch.from_numpy(img), torch.from_numpy(kps)


if __name__ == '__main__':
    from config import config as cfg

    train_set = KeyPointDataset(cfg, label_txt=cfg.train_list, mode="train")
    train_loader = DataLoader(train_set, batch_size=1, num_workers=2,
                              shuffle=True, drop_last=False, collate_fn=train_set.collate_fn)
    for batch_idx, (img, hp, masks, kpts) in enumerate(train_loader):
        print(img.size(), hp.size(), masks.size(), kpts.size())
        print(kpts.size())
        print(type(kpts))
        print(kpts.tolist())
        print(type(kpts.tolist()))
        exit()

