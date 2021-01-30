# -*- coding: utf-8 -*-
# @Time    : 2020/12/28 21:02
# @Author  : Zeqi@@
# @FileName: utils.py
# @Software: PyCharm

import os
import cv2
from PIL import Image
import numpy as np

class Data_augmentation():
    """
    Resize, crop, HSV augmentation
    """
    def __init__(self,
                 input_shape,
                 max_boxes=100,
                 jitter=.3,
                 hue=.1,
                 sat=1.5,
                 val=1.5,
                 visual=True):

        self.h, self.w = input_shape
        self.max_boxes = max_boxes
        self.hue = hue
        self.sat = sat
        self.val = val
        self.jitter = jitter
        self.visual = visual

    def resize_image(self, image):
        # 对图像进行缩放并且进行长和宽的扭曲
        new_ar = self.w / self.h * np.random.uniform(1 - self.jitter, 1 + self.jitter) / np.random.uniform(1 - self.jitter, 1 + self.jitter)
        _scale = np.random.rand() * 2
        scale = 0.5 if _scale < 0.5 else _scale
        if new_ar < 1:
            nh = int(scale * self.h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * self.w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)
        return image, nh, nw

    def place_image_in_input(self, image, nh, nw):
        # 将图像多余的部分加上灰条
        # 以给定的形状创建一个数组，并在数组中加入在[0,1]之间均匀分布的随机样本 rand()
        dx = int(np.random.rand() * (self.w - nw))
        dy = int(np.random.rand() * (self.h - nh))
        new_image = Image.new('RGB', (self.w, self.h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image = new_image
        return image, dx, dy

    def HSV_augmentation(self, image):
        hue = np.random.uniform(-self.hue, self.hue)
        sat = np.random.uniform(1, self.sat) if np.random.rand() < .5 else 1 / np.random.uniform(1, self.sat)
        val = np.random.uniform(1, self.val) if np.random.rand() < .5 else 1 / np.random.uniform(1, self.val)

        x = cv2.cvtColor(np.array(image, np.float32) / 255, cv2.COLOR_RGB2HSV)
        # 0 通道为 Hua
        x[..., 0] += hue * 360  # 取最后一个通道的x 举例 x.shape （2， 4， 3， 5） - x[..., 0] - x[..., 0].shape (2, 4, 3)
        x[..., 0][x[..., 0] > 1] -= 1  # 所有x[...,0]中大于1的数减1
        x[..., 0][x[..., 0] < 0] += 1  # 所有x[...,0]中小于1的数加1
        x[..., 1] *= sat  # 所有x[...,1]中数*sat
        x[..., 2] *= val  # 所有x[...,2]中数*val
        x[x[:, :, 0] > 360, 0] = 360  # 将 0通道中大于360的数设为360
        x[:, :, 1:][x[:, :, 1:] > 1] = 1  # 将 1， 2通道中大于1的数设为1
        x[x < 0] = 0  # 将 所有小于0的数设为0
        RGB_image = cv2.cvtColor(x, cv2.COLOR_HSV2RGB)  # numpy array, 0 to 1
        return RGB_image

    def finetune_bbox(self, box, nh, nw, dy, dx, flip):
        # 将box进行调整
        box_data = np.zeros((self.max_boxes, 5))
        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]] * nw / self.iw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / self.ih + dy
            if flip: box[:, [0, 2]] = self.w - box[:, [2, 0]]
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > self.w] = self.w
            box[:, 3][box[:, 3] > self.h] = self.h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box
            if len(box) > self.max_boxes: box = box[:self.max_boxes]
            box_data[:len(box)] = box
        return box_data

    def main_train(self, annotation_line):
        line = annotation_line.split()
        image = Image.open(line[0])
        bbox = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

        self.iw, self.ih = image.size  # PIL image
        resized_image, nh, nw = self.resize_image(image)
        new_image, dx, dy = self.place_image_in_input(resized_image, nh, nw)

        # 翻转图像
        flip = np.random.rand() < .5
        if flip: new_image = new_image.transpose(Image.FLIP_LEFT_RIGHT)

        # 色彩增强
        aug_image = self.HSV_augmentation(new_image)

        # 将box进行调整
        aug_box = self.finetune_bbox(bbox, nh, nw, dy, dx, flip)

        if self.visual:
            image_data = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)

            for box in aug_box:
                box = [int(b) for b in box]
                cv2.rectangle(image_data, (box[0], box[1]), (box[2], box[3]), color=(255, 255, 255), thickness=1)

            cv2.imshow('Image', image_data)
            cv2.waitKey(1000)

        return aug_image, aug_box

    def main_val(self, annotation_line):
        line = annotation_line.split()
        image = Image.open(line[0])
        box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

        self.iw, self.ih = image.size  # PIL image

        # 对图像进行缩放
        nh = int(self.h)
        nw = int(self.w)
        image = image.resize((nw, nh), Image.BICUBIC)

        # 将图像多余的部分加上灰条
        dx = int((self.w - nw))
        dy = int((self.h - nh))
        new_image = Image.new('RGB', (self.w, self.h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image = new_image

        # 将box进行调整
        box_data = np.zeros((self.max_boxes, 5))
        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]] * nw / self.iw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / self.ih + dy
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > self.w] = self.w
            box[:, 3][box[:, 3] > self.h] = self.h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box
            if len(box) > self.max_boxes: box = box[:self.max_boxes]
            box_data[:len(box)] = box

        # x = cv2.cvtColor(np.array(image, np.float32) / 255, cv2.COLOR_RGB2HSV)
        image = np.array(image, dtype=np.float32) / 255.

        if self.visual:
            image_data = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            for box in box_data:
                box = [int(b) for b in box]
                cv2.rectangle(image_data, (box[0], box[1]), (box[2], box[3]), color=(255, 255, 255), thickness=1)

            cv2.imshow('Image', image_data)
            cv2.waitKey(1000)

        return image, box_data

def get_random_data(annotation_line, input_shape, max_boxes=100, jitter=.3, hue=.1, sat=1.5, val=1.5, visual=True):
    '''random preprocessing for real-time data augmentation'''
    line = annotation_line.split()
    image = Image.open(line[0])
    iw, ih = image.size
    h, w = input_shape
    box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

    # 对图像进行缩放并且进行长和宽的扭曲  0.7 1.3
    # new_ar = w / h * rand(1 - jitter, 1 + jitter) / rand(1 - jitter, 1 + jitter) # -2~4
    new_ar = w/h*np.random.uniform(1 - jitter, 1 + jitter)/np.random.uniform(1 - jitter, 1 + jitter)
    _scale = np.random.rand()*2
    scale  = 0.25 if _scale < 0.25 else _scale
    if new_ar < 1:
        nh = int(scale * h)
        nw = int(nh * new_ar)
    else:
        nw = int(scale * w)
        nh = int(nw / new_ar)
    image = image.resize((nw, nh), Image.BICUBIC)

    # 将图像多余的部分加上灰条
    # 以给定的形状创建一个数组，并在数组中加入在[0,1]之间均匀分布的随机样本 rand()
    # dx = int(rand(0, w - nw))
    # dy = int(rand(0, h - nh))
    dx = int(np.random.rand() * (w - nw))
    dy = int(np.random.rand() * (h - nh))
    new_image = Image.new('RGB', (w, h), (128, 128, 128))
    new_image.paste(image, (dx, dy))
    image = new_image

    # 翻转图像
    flip = np.random.rand() < .5
    if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

    # 色域扭曲
    # hue = np.random.rand(-hue, hue)
    # sat = np.random.rand(1, sat) if np.random.rand() < .5 else 1 / np.random.rand(1, sat)
    # val = np.random.rand(1, val) if np.random.rand() < .5 else 1 / np.random.rand(1, val)

    hue = np.random.uniform(-hue, hue)
    sat = np.random.uniform(1, sat) if np.random.rand() < .5 else 1 / np.random.uniform(1, sat)
    val = np.random.uniform(1, val) if np.random.rand() < .5 else 1 / np.random.uniform(1, val)

    x = cv2.cvtColor(np.array(image, np.float32) / 255, cv2.COLOR_RGB2HSV)
    # 0 通道为 Hua
    x[..., 0] += hue * 360              # 取最后一个通道的x 举例 x.shape （2， 4， 3， 5） - x[..., 0] - x[..., 0].shape (2, 4, 3)
    x[..., 0][x[..., 0] > 1] -= 1       # 所有x[...,0]中大于1的数减1
    x[..., 0][x[..., 0] < 0] += 1       # 所有x[...,0]中小于1的数加1
    x[..., 1] *= sat                    # 所有x[...,1]中数*sat
    x[..., 2] *= val                    # 所有x[...,2]中数*val
    x[x[:, :, 0] > 360, 0] = 360        # 将 0通道中大于360的数设为360
    x[:, :, 1:][x[:, :, 1:] > 1] = 1    # 将 1， 2通道中大于1的数设为1
    x[x < 0] = 0                        # 将 所有小于0的数设为0
    image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB)  # numpy array, 0 to 1

    # 将box进行调整
    box_data = np.zeros((max_boxes, 5))
    if len(box) > 0:
        np.random.shuffle(box)
        box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
        box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
        if flip: box[:, [0, 2]] = w - box[:, [2, 0]]
        box[:, 0:2][box[:, 0:2] < 0] = 0
        box[:, 2][box[:, 2] > w] = w
        box[:, 3][box[:, 3] > h] = h
        box_w = box[:, 2] - box[:, 0]
        box_h = box[:, 3] - box[:, 1]
        box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box
        if len(box) > max_boxes: box = box[:max_boxes]
        box_data[:len(box)] = box

    if visual:
        for box in box_data:
            box = [int(b) for b in box]
            cv2.rectangle(image_data, (box[0], box[1]), (box[2], box[3]), color=(255, 255, 255), thickness=1)

        image_data = cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR)
        cv2.imshow('Image', image_data)
        cv2.waitKey(1000)

    return image_data, box_data

if __name__ == "__main__":
    annotation_path =  '../Preparation/data_txt'
    annotation_lines = open(os.path.join(annotation_path, 'kitti_ssd_obj_trainval.txt')).readlines()
    print(len(annotation_lines))

    # for annotation_line in annotation_lines:
    #     image_data, box_data = get_random_data(annotation_line,
    #                                            input_shape=(160, 480),
    #                                            max_boxes=100,
    #                                            jitter=.3,
    #                                            hue=.1,
    #                                            sat=1.5,
    #                                            val=1.5,
    #                                            visual = True)

    data_aug = Data_augmentation(input_shape=(160, 480),
                                 max_boxes=100,
                                 jitter=.3,
                                 hue=.1,
                                 sat=1.5,
                                 val=1.5,
                                 visual=False)

    for annotation_line in annotation_lines:

        image, box_data = data_aug.main_val(annotation_line)
        print("Image: {}, min: {}, max: {}".format(np.shape(image), np.min(image), np.max(image)))
        print("Box_Data: {}".format(np.shape(box_data)))
        print(" ")

