# -*- coding: utf-8 -*-

class config:
    # 数据集
    train_list = "data/train/label.txt"
    test_list = "data/test/label.txt"
    num_workers = 4
    batch_size = 32
    manualseed = 0

    # 数据处理参数
    mu = [0.485, 0.456, 0.406]
    sigma = [0.229, 0.224, 0.225]
    hm_stride = 4
    hm_alpha = 100
    img_size = 480
    hm_sigma = img_size / hm_stride / 16.
    img_hp_rate = 4

    # 网络
    pretrained = r"checkout/resnet50.pth"
    num_kps = 4
    checkoutpth = ""

    # 训练参数
    num_epochs = 500
    start_epoch = 1
    lr = 1e-3
    beta1 = 0.5
    beta2 = 0.999
    stones = [10, 30, 50, 80]
    interval_print = 20
    interval_save = 10
    checkout = "checkout"
    logs = checkout + "/logs"
    is_best = 0.
