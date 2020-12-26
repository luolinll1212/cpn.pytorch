# -*- coding: utf-8 -*-
from config import config as cfg

def compute_model_kps(out):
    # 预测点，三维
    pred_kps = []
    for i in range(out.size(0)):
        map = out[0,...]
        point = []
        for h in map:
            pos = h.view(-1).argmax().item()
            x = (pos % h.size(1)) * cfg.img_hp_rate
            y = (pos // h.size(1)) * cfg.img_hp_rate
            point += [(x, y)]
        pred_kps += [(point)]
    return pred_kps


