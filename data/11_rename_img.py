# -*- coding: utf-8 -*-
import os
import json
import numpy as np
import random
import shutil
import datetime, time, random, hashlib

def check_dir(path):
    for file in os.listdir(path):
        full_path = os.path.join(path, file)
        print("delete --- ", full_path)
        os.remove(full_path)

def make_new_name():
    # 获取当前时间
    time.sleep(0.01)
    new_time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S_%f')
    # 产生随机数
    new_round = str(random.uniform(0,1000))
    # 组合新的字符串
    new_str = (new_time + new_round).encode("utf-8")
    # md5
    md5 = hashlib.md5()
    md5.update(new_str)
    return md5.hexdigest()


def get_label_json(json_path):

    data = json.load(open(json_path, 'r'))

    # 拿到坐标
    mark_x1 = data["mark_x1"] # 左上
    mark_y1 = data["mark_y1"]
    mark_x2 = data["mark_x2"] # 右上
    mark_y2 = data["mark_y2"]
    mark_x3 = data["mark_x3"] # 左下
    mark_y3 = data["mark_y3"]
    mark_x4 = data["mark_x4"] # 右下
    mark_y4 = data["mark_y4"]


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

    return dict_label, data['name']

def gen_gt(gt, path):

    # 将图片和标签拷贝到gt
    for file in os.listdir(path):
        full_path = path + "/" + file

        # 拿到图片 -> 拷贝
        if full_path.endswith(".json"):
            json_path = full_path     # json path
            print(json_path)

            # 定义label和name
            dict_label, img_name = get_label_json(json_path)

            # 图片名字
            old_img_path = os.path.join(path, img_name)
            # 新的名字
            new_seed = make_new_name()
            new_img_name = new_seed + ".jpg"
            new_json_name = new_seed + ".json"
            dict_label["name"] =  new_img_name
            new_img_path = gt + "/" + new_img_name

            # 新的json名字
            new_json_path = gt + "/" + new_json_name

            # 保存json
            json_str = json.dumps(dict_label)
            with open(new_json_path, 'w') as json_file:
                json_file.write(json_str)

            # 拷贝图片
            shutil.copyfile(old_img_path, new_img_path)




# 标签合并，不改变图片大小
if __name__ == '__main__':
    path = r"gt_12.3_all"
    gt = r"gt"

    # 生成gt文件
    if not os.path.exists(gt):
        os.mkdir(gt)
    check_dir(gt)

    # 产生标签文件
    gen_gt(gt, path)
