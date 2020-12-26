# -*- coding: utf-8 -*-
import os

def remove_no_label_img(path):
    img_list = []
    for file in os.listdir(path):
        full_path = os.path.join(path, file)
        if full_path.endswith(".jpg"):
            img_list += [(full_path)]

    # 根据标签删除图片
    for img in img_list:
        json_file = img.replace(".jpg", ".json")
        if not os.path.exists(json_file): # 标签文件不存在
            print("delete ------ ",img)
            os.remove(img)

    print(len(img_list))


if __name__ == '__main__':
    path = r"D:\work\PycharmProjects\keypoint02\data\gt_12.1"

    remove_no_label_img(path)