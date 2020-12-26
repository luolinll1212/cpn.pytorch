# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def check_label(line):
    # 读取每行数据
    line = line.strip().split()

    # 读取图片
    img_path = line[0]
    image = np.array(Image.open(img_path), dtype=np.float32) / 255.  # 灰度图片

    # 拿到所有关键点
    points = line[1:]
    landmask = []
    for i in range(4):
        x = points[i * 2]
        y = points[i * 2 + 1]
        landmask += [(x, y)]
    landmask = np.array(landmask, dtype=np.float32)

    # 显示
    plt.figure()
    plt.title("gt")
    plt.imshow(image, cmap='gray')
    plt.scatter(landmask[:, 0], landmask[:, 1])
    plt.show()



if __name__ == '__main__':
    line = "train_data/imgs/1041.jpg 49.23 40.65 190.98 43.14 70.58 212.59 167.53 210.1"

    # 查看图片
    check_label(line)


