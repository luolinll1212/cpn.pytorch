# -*- coding: utf-8 -*-
import os, json
import cv2 as cv

font = cv.FONT_HERSHEY_COMPLEX

def check_dir(path):
    for file in os.listdir(path):
        full_path = os.path.join(path, file)
        print("delete - {}".format(full_path))
        os.remove(full_path)

def new_pixel(x, y, w, h, size):
    new_x = x * size / w
    new_y = y * size / h
    return new_x, new_y

def gen_data(path, save_path):

    # 读取文件
    for file in os.listdir(path):
        full_path = path +"/" + file

        if full_path.endswith(".json"):
            json_path = full_path
            data_label = json.load(open(json_path, 'r'))

            # 拿到特征点坐标
            mark_x1, mark_y1 = data_label["mark_x1"], data_label["mark_y1"]  # 左上
            mark_x2, mark_y2 = data_label["mark_x2"], data_label["mark_y2"]  # 右上
            mark_x3, mark_y3 = data_label["mark_x3"], data_label["mark_y3"]  # 左下
            mark_x4, mark_y4 = data_label["mark_x4"], data_label["mark_y4"]  # 右下

            # # 拿到坐标
            # for obj in data_label['shapes']:
            #     if obj['label'] == "left_up":  # 左上点
            #         mark_x1 = int(obj['points'][0][0])
            #         mark_y1 = int(obj['points'][0][1])
            #     if obj['label'] == "right_up":  # 右上点
            #         mark_x2 = int(obj['points'][0][0])
            #         mark_y2 = int(obj['points'][0][1])
            #     if obj['label'] == "left_down":  # 左下点
            #         mark_x3 = int(obj['points'][0][0])
            #         mark_y3 = int(obj['points'][0][1])
            #     if obj['label'] == "right_down":  # 右下点
            #         mark_x4 = int(obj['points'][0][0])
            #         mark_y4 = int(obj['points'][0][1])

           # 读取图片
           #  img_path = os.path.join(path, data_label['imagePath'])
            img_path = os.path.join(path, data_label['name'])
            print(img_path)
            image = cv.imread(img_path)
            h, w, c = image.shape

            size = 512

            # 压缩图片
            new_image = cv.resize(image, (size, size), interpolation=cv.INTER_AREA)
            new_mark_x1, new_mark_y1 = new_pixel(mark_x1, mark_y1, w, h, size)
            new_mark_x2, new_mark_y2 = new_pixel(mark_x2, mark_y2, w, h, size)
            new_mark_x3, new_mark_y3 = new_pixel(mark_x3, mark_y3, w, h, size)
            new_mark_x4, new_mark_y4 = new_pixel(mark_x4, mark_y4, w, h, size)

            # 检查
            # 左上 left_up
            cv.circle(new_image, (int(new_mark_x1), int(new_mark_y1)), 3, (255, 0, 0), -1)
            cv.putText(new_image, "left_up", (int(new_mark_x1), int(new_mark_y1)), font, 0.5, (255, 0, 0))
            # 右上 right_up
            cv.circle(new_image, (int(new_mark_x2), int(new_mark_y2)), 3, (0, 255, 0), -1)
            cv.putText(new_image, "right_up", (int(new_mark_x2), int(new_mark_y2)), font, 0.5, (0, 255, 0))
            # 左下 left_down
            cv.circle(new_image, (int(new_mark_x3), int(new_mark_y3)), 3, (0, 255, 255), -1)
            cv.putText(new_image, "left_down", (int(new_mark_x3), int(new_mark_y3)), font, 0.5, (0, 255, 255))
            # 右下 right_down
            cv.circle(new_image, (int(new_mark_x4), int(new_mark_y4)), 3, (255, 255, 0), -1)
            cv.putText(new_image, "right_down", (int(new_mark_x4), int(new_mark_y4)), font, 0.5, (255, 255, 0))

            # new_image_name = save_path + "/" + data_label['imagePath']
            new_image_name = save_path + "/" + data_label['name']

            cv.imwrite(new_image_name, new_image, [int(cv.IMWRITE_JPEG_QUALITY), 95])


if __name__ == '__main__':
    gt = r"gt"
    save_path = r"check_img"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    check_dir(save_path)

    gen_data(gt, save_path)

