# -*- coding: utf-8 -*-
import os
import json
import numpy as np
import random
import shutil
import datetime, time

def check_dir(path):
    for file in os.listdir(path):
        full_path = os.path.join(path, file)
        print("delete --- ", full_path)
        os.remove(full_path)

def make_new_name():
    # 产生随机数
    r1 = random.uniform(0, 10000)
    r2 = random.uniform(0, 10000)
    r3 = random.uniform(0, 10000)
    r4 = random.uniform(0, 10000)

    # 拿到整数部分
    a1 = int(r1 * 1e+4)
    a2 = int(r2 * 1e+4)
    a3 = int(r3 * 1e+4)
    a4 = int(r4 * 1e+4)

    # 计算总和
    sum = a1 - a2 + a3 - a4

    return str(sum)

def make_new_name2():
    time.sleep(0.01)
    new_name = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S_%f')
    return new_name


def get_label_json(json_path):

    data = json.load(open(json_path, 'r'))

    # 拿到坐标
    for obj in data['shapes']:
        if obj['label'] == "left_up":  # 左上点
            mark_x1 = int(obj['points'][0][0])
            mark_y1 = int(obj['points'][0][1])
        if obj['label'] == "right_up":  # 右上点
            mark_x2 = int(obj['points'][0][0])
            mark_y2 = int(obj['points'][0][1])
        if obj['label'] == "left_down":  # 左下点
            mark_x3 = int(obj['points'][0][0])
            mark_y3 = int(obj['points'][0][1])
        if obj['label'] == "right_down":  # 右下点
            mark_x4 = int(obj['points'][0][0])
            mark_y4 = int(obj['points'][0][1])

    dict_label = {}
    # 向字典写值
    dict_label["mark_x1"] = mark_x1
    dict_label["mark_y1"] = mark_y1
    dict_label["mark_x2"] = mark_x2
    dict_label["mark_y2"] = mark_y2
    dict_label["mark_x3"] = mark_x3
    dict_label["mark_y3"] = mark_y3
    dict_label["mark_x4"] = mark_x4
    dict_label["mark_y4"] = mark_y4

    return dict_label, data['imagePath']


def gen_gt(gt, path):
    # 创建gt文件夹
    if not os.path.exists(gt):
        os.mkdir(gt)

    # 将图片和标签拷贝到gt
    for dir_path in path:
        for file in os.listdir(dir_path):
            full_path = dir_path + "/" + file

            if full_path.endswith(".json"):
                json_path = full_path     # json path
                print(json_path)

                dict_label, img_name = get_label_json(json_path)

                # 图片名字
                old_img_path = os.path.join(dir_path, img_name)
                seed = make_new_name2()
                dict_label["name"] =  seed + ".jpg"
                new_img_path = gt + "/" + seed + ".jpg"

                # 新的json名字
                new_json_path = gt + "/" + seed + ".json"

                # 保存json
                json_str = json.dumps(dict_label)
                with open(new_json_path, 'w') as json_file:
                    json_file.write(json_str)

                # 拷贝图片
                shutil.copyfile(old_img_path, new_img_path)



# 标签合并，不改变图片大小
if __name__ == '__main__':
    path = [
            # r"D:\work\PycharmProjects\keypoint02\data\gt_11.13"
            # r"D:\work\PycharmProjects\keypoint02\data\gt_12.1",
            r"D:\work\PycharmProjects\keypoint02\data\gt_12.3"
           ]
    gt = r"D:\save\gt-01"

    # check_dir(gt)

    gen_gt(gt, path)
