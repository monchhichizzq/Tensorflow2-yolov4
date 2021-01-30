# -*- coding: utf-8 -*-
# @Time    : 2020/12/28 23:33
# @Author  : Zeqi@@
# @FileName: mosaic_utils.py
# @Software: PyCharm

import os
import cv2
from PIL import Image
import numpy as np


def merge_bboxes(bboxes, cutx, cuty):
    merge_bbox = []
    for i in range(len(bboxes)):
        for box in bboxes[i]:
            tmp_box = []
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

            if i == 0:
                if y1 > cuty or x1 > cutx:
                    continue
                if y2 >= cuty and y1 <= cuty:
                    y2 = cuty
                    if y2 - y1 < 5:
                        continue
                if x2 >= cutx and x1 <= cutx:
                    x2 = cutx
                    if x2 - x1 < 5:
                        continue

            if i == 1:
                if y2 < cuty or x1 > cutx:
                    continue

                if y2 >= cuty and y1 <= cuty:
                    y1 = cuty
                    if y2 - y1 < 5:
                        continue

                if x2 >= cutx and x1 <= cutx:
                    x2 = cutx
                    if x2 - x1 < 5:
                        continue

            if i == 2:
                if y2 < cuty or x2 < cutx:
                    continue

                if y2 >= cuty and y1 <= cuty:
                    y1 = cuty
                    if y2 - y1 < 5:
                        continue

                if x2 >= cutx and x1 <= cutx:
                    x1 = cutx
                    if x2 - x1 < 5:
                        continue

            if i == 3:
                if y1 > cuty or x2 < cutx:
                    continue

                if y2 >= cuty and y1 <= cuty:
                    y2 = cuty
                    if y2 - y1 < 5:
                        continue

                if x2 >= cutx and x1 <= cutx:
                    x1 = cutx
                    if x2 - x1 < 5:
                        continue

            tmp_box.append(x1)
            tmp_box.append(y1)
            tmp_box.append(x2)
            tmp_box.append(y2)
            tmp_box.append(box[-1])
            merge_bbox.append(tmp_box)
    return merge_bbox

class Data_augmentation_with_Mosaic():
    def __init__(self,
                 four_annotation_lines,
                 input_shape,
                 max_boxes=100,
                 hue=.1,
                 sat=1.5,
                 val=1.5,
                 visual=True):

        self.four_annotation_lines = four_annotation_lines

        self.h, self.w = input_shape
        self.min_offset_x = 0.4
        self.min_offset_y = 0.4
        self.scale_low = 1 - min(self.min_offset_x, self.min_offset_y)
        self.scale_high = self.scale_low + 0.2

        self.max_boxes = max_boxes
        self.hue = .1
        self.sat = 1.5
        self.val = 1.5

        self.place_x = [0, 0, int(self.w * self.min_offset_x), int(self.w * self.min_offset_x)]
        self.place_y = [0, int(self.h * self.min_offset_y), int(self.h * self.min_offset_y), 0]

        self.visual = visual


    def main(self):
        image_datas = []
        box_datas = []
        index = 0

        for annotation_line in self.four_annotation_lines:
            # 每一行进行分割
            line_content = annotation_line.split()
            # 打开图片
            image = Image.open(line_content[0])
            image = image.convert("RGB")
            # 图片的大小
            iw, ih = image.size
            # 保存框的位置
            box = np.array([np.array(list(map(int, box.split(',')))) for box in line_content[1:]])

            # 是否翻转图片
            flip = np.random.rand() < .5
            if flip and len(box) > 0:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                box[:, [0, 2]] = iw - box[:, [2, 0]]

            # 对输入进来的图片进行缩放
            new_ar = self.w / self.h
            scale = np.random.uniform(self.scale_low, self.scale_high)
            if new_ar < 1:
                nh = int(scale * self.h)
                nw = int(nh * new_ar)
            else:
                nw = int(scale * self.w)
                nh = int(nw / new_ar)
            image = image.resize((nw, nh), Image.BICUBIC)

            # 进行色域变换
            hue = np.random.uniform(-self.hue, self.hue)
            sat = np.random.uniform(1, self.sat) if np.random.uniform() < .5 else 1 / np.random.uniform(1, self.sat)
            val = np.random.uniform(1, self.val) if np.random.uniform() < .5 else 1 / np.random.uniform(1, self.val)
            x = cv2.cvtColor(np.array(image, np.float32) / 255, cv2.COLOR_RGB2HSV)
            x[..., 0] += hue * 360
            x[..., 0][x[..., 0] > 1] -= 1
            x[..., 0][x[..., 0] < 0] += 1
            x[..., 1] *= sat
            x[..., 2] *= val
            x[x[:, :, 0] > 360, 0] = 360
            x[:, :, 1:][x[:, :, 1:] > 1] = 1
            x[x < 0] = 0
            image = cv2.cvtColor(x, cv2.COLOR_HSV2RGB)  # numpy array, 0 to 1

            image = Image.fromarray((image * 255).astype(np.uint8))
            # 将图片进行放置，分别对应四张分割图片的位置
            dx = self.place_x[index]
            dy = self.place_y[index]


            new_image = Image.new('RGB', (self.w, self.h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image) / 255

            if self.visual:
                image_ = np.array(image_data * 255., dtype=np.uint8)
                # print(np.shape(image_))
                image_ = cv2.cvtColor(image_, cv2.COLOR_RGB2BGR)
                cv2.imshow('s Image', image_)
                cv2.waitKey(100)

            index = index + 1
            box_data = []
            # 对box进行重新处理
            if len(box) > 0:
                np.random.shuffle(box)
                box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
                box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
                box[:, 0:2][box[:, 0:2] < 0] = 0
                box[:, 2][box[:, 2] > self.w] = self.w
                box[:, 3][box[:, 3] > self.h] = self.h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w > 1, box_h > 1)]
                box_data = np.zeros((len(box), 5))
                box_data[:len(box)] = box

            image_datas.append(image_data)
            box_datas.append(box_data)

        # 将图片分割，放在一起
        cutx = np.random.randint(int(self.w * self.min_offset_x), int(self.w * (1 - self.min_offset_x)))
        cuty = np.random.randint(int(self.h * self.min_offset_y), int(self.h * (1 - self.min_offset_y)))

        new_image = np.zeros([self.h, self.w, 3])
        # print("\n mosaic")
        # print('image datas',np.shape(image_datas), cutx, cuty)
        new_image[:cuty, :cutx, :] = image_datas[0][:cuty, :cutx, :]
        new_image[cuty:, :cutx, :] = image_datas[1][cuty:, :cutx, :]
        new_image[cuty:, cutx:, :] = image_datas[2][cuty:, cutx:, :]
        new_image[:cuty, cutx:, :] = image_datas[3][:cuty, cutx:, :]

        # 对框进行进一步的处理
        new_boxes = merge_bboxes(box_datas, cutx, cuty)

        # 将box进行调整
        box_data = np.zeros((self.max_boxes, 5))
        if len(new_boxes) > 0:
            if len(new_boxes) > self.max_boxes: new_boxes = new_boxes[:self.max_boxes]
            box_data[:len(new_boxes)] = new_boxes

        if self.visual:

            # print(new_image.shape, np.max(new_image), np.min(new_image))
            new_image = np.array(new_image * 255., dtype=np.uint8)
            # print(new_image.shape, np.max(new_image), np.min(new_image))

            for box in box_data:
                box = [int(b) for b in box]
                cv2.rectangle(new_image, (box[0], box[1]), (box[2], box[3]), color=(255, 255, 255), thickness=1)

            new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
            cv2.imshow('Image', new_image)
            cv2.waitKey(1000)

        return new_image, box_data


if __name__ == "__main__":
    annotation_path =  '../Preparation/data_txt'
    annotation_lines = open(os.path.join(annotation_path, 'kitti_ssd_obj_test.txt')).readlines()
    # print(len(annotation_lines))

    four_annotation_lines = []
    for i, line in enumerate(annotation_lines):
        four_annotation_lines.append(line)
        if (i+1) % 4==0:
            mosaic_aug = Data_augmentation_with_Mosaic(four_annotation_lines,
                                                         input_shape=(320, 960),
                                                         max_boxes=100,
                                                         hue=.1,
                                                         sat=1.5,
                                                         val=1.5,
                                                         visual=True)
            image_data, box_data = mosaic_aug.main()
            four_annotation_lines = []
