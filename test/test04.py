# -*- coding: utf-8 -*-
import torch
import numpy as np


x = torch.tensor([1, 2, np.nan])

if torch.isnan(x):
    print("aa")
else:
    print("bb")



