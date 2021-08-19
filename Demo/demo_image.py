# -*- coding: utf-8 -*-
# @Time    : 2021/1/9 23:37
# @Author  : Zeqi@@
# @FileName: demo_image.py
# @Software: PyCharm

import os
import sys
sys.path.append(os.path.join(os.getcwd(), ".."))

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from yolo_predictor import YOLO

yolo = YOLO()

image_dir = '/Users/babalia/Desktop/Applications/object_detection/yolov5_samples/data/voc/test/VOCdevkit/VOC2007/JPEGImages'

for image in os.listdir(image_dir):
    img_path = os.path.join(image_dir, image)
    print(img_path)
    image = Image.open(img_path)
    r_image = yolo.detect_image(image)

    num_image = np.array(r_image)
    bgr_image = cv2.cvtColor(num_image, cv2.COLOR_RGB2BGR)
    cv2.imshow('demo', bgr_image)
    cv2.waitKey(1000)


