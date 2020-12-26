# -*- coding: utf-8 -*-
import os, json
import numpy as np
import cv2 as cv

np.random.seed(0)


def check_path(path):
    for file in os.listdir(path):
        full_path = os.path.join(path, file)
        print("delete - {}".format(full_path))
        os.remove(full_path)


def check_file(label_txt):
    if os.path.exists(label_txt):
        os.remove(label_txt)
    fp = open(label_txt, "w", encoding="utf-8")
    fp.close()


def new_pixel(x, y, w, h, size):
    new_x = x * size / w
    new_y = y * size / h
    return new_x, new_y


def data_partition(length, rate):
    # 划分数据集
    seed = np.random.randint(0, length, size=(int(length * rate)))
    seed = seed.tolist()

    # 选择数据集
    train_index = []
    test_index = []
    for i in range(length):
        if i not in seed:
            train_index += [(i)]
        else:
            test_index += [(i)]
    return train_index, test_index


def write_data(json_list, index, save_img_path, save_label_txt):
    # 打开标签文件
    with open(save_label_txt, "w+", encoding="utf-8") as fwrite:
        # 读取json文件，写入数据
        for i in index:
            json_path = json_list[i]  # json文件
            print("deal with ----- ", json_path)
            data_label = json.load(open(json_path, 'r'))

            # 拿到特征点坐标
            mark_x1, mark_y1 = data_label["mark_x1"], data_label["mark_y1"]   # 左上
            mark_x2, mark_y2 = data_label["mark_x2"], data_label["mark_y2"]  # 右上
            mark_x3, mark_y3 = data_label["mark_x3"], data_label["mark_y3"]  # 左下
            mark_x4, mark_y4 = data_label["mark_x4"], data_label["mark_y4"]  # 右下


            # 读取图片
            img_path = json_path.replace(".json", ".jpg")
            image = cv.imread(img_path)

            h, w, c = image.shape

            # # 压缩图片
            # new_image = cv.resize(image, (size, size), interpolation=cv.INTER_AREA)
            # new_image_gray = cv.cvtColor(new_image, cv.COLOR_BGR2GRAY)
            # new_mark_x1, new_mark_y1 = new_pixel(mark_x1, mark_y1, w, h, size)
            # new_mark_x2, new_mark_y2 = new_pixel(mark_x2, mark_y2, w, h, size)
            # new_mark_x3, new_mark_y3 = new_pixel(mark_x3, mark_y3, w, h, size)
            # new_mark_x4, new_mark_y4 = new_pixel(mark_x4, mark_y4, w, h, size)

            # # 检查
            # cv.circle(new_image_gray, (int(new_mark_x1), int(new_mark_y1)), 3, (255, 0, 0), -1)
            # cv.circle(new_image_gray, (int(new_mark_x2), int(new_mark_y2)), 3, (0, 255, 0), -1)
            # cv.circle(new_image_gray, (int(new_mark_x3), int(new_mark_y3)), 3, (0, 0, 255), -1)
            # cv.circle(new_image_gray, (int(new_mark_x4), int(new_mark_y4)), 3, (0, 255, 255), -1)
            # cv.imshow("new_image_gray", new_image_gray)
            # cv.waitKey(0)
            # cv.destroyAllWindows()

            # 保存图片
            label_img = save_img_path + "/" + data_label['name']
            # 保存点
            landmask = "{} {} {} {} {} {} {} {}".format(round(mark_x1, 2), round(mark_y1, 2),
                                                        round(mark_x2, 2), round(mark_y2, 2),
                                                        round(mark_x3, 2), round(mark_y3, 2),
                                                        round(mark_x4, 2), round(mark_y4, 2))
            line = "{} {}\n".format(label_img, landmask)

            fwrite.write(line)
            cv.imwrite(f"{save_img_path}/{data_label['name']}", image, [int(cv.IMWRITE_JPEG_QUALITY), 95])


def gen_data(gt, train_img_path, train_label_txt, test_img_path, test_label_txt):
    # 拿到所有图片的路径
    json_list = [os.path.join(gt, file) for file in os.listdir(gt) if file.endswith(".json")]

    # 划分训练集和测试集
    rate = 0.05
    train_index, test_index = data_partition(len(json_list), rate)

    # 训练集写入
    write_data(json_list, train_index, train_img_path, train_label_txt)

    # 测试集写入
    write_data(json_list, test_index, test_img_path, test_label_txt)


if __name__ == '__main__':
    gt = "data/gt"

    # 训练集
    train = "data/train"
    train_img_path = train + "/imgs"  # 保存图片位置
    if not os.path.exists(train):
        os.mkdir(train)
        os.mkdir(train_img_path)
    train_label_txt = train + "/label.txt"

    # 测试集
    test = "data/test"
    test_img_path = test + "/imgs"  # 保存图片位置
    if not os.path.exists(test):
        os.mkdir(test)
        os.mkdir(test_img_path)
    test_label_txt = test + "/label.txt"

    # 检查训练集
    check_path(train_img_path)
    check_file(train_label_txt)

    # 检查测试集
    check_path(test_img_path)
    check_file(test_label_txt)

    # 生成标签
    gen_data(gt, train_img_path, train_label_txt, test_img_path, test_label_txt)

