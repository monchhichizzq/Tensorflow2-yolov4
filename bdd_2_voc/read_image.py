# -*- coding: utf-8 -*-
# @Time    : 2021/1/29 22:34
# @Author  : Zeqi@@
# @FileName: read_image.py
# @Software: PyCharm

import os
import cv2
import numpy as np
from tqdm import tqdm

path = 'D:/BDD100K/images/100k/train'

for frame_id in tqdm(os.listdir(path)):
    frame_path = os.path.join(path, frame_id)
    frame = cv2.imread(frame_path)
    cv2.imshow('frame', frame) #  (720, 1280, 3)
    cv2.waitKey(1)

