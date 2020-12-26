# -*- coding: utf-8 -*-
from PIL import Image, ImageDraw
import numpy as np


blank = Image.new("RGB", [635, 800], "white")

drawObject = ImageDraw.Draw(blank)
drawObject.ellipse((150, 50, 500, 450), fill="green")
iar = np.asarray(blank)
iar.flags.writeable = True
for i in range(400, 650):
    for j in range(280, 370):
        iar[i][j] = [0, 128, 0]
img1 = Image.fromarray(iar)
# img1.show()
img1.save("ovaltree.png", "PNG")