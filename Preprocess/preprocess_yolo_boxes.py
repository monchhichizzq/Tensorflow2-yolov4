# -*- coding: utf-8 -*-
# @Time    : 2020/12/29 17:19
# @Author  : Zeqi@@
# @FileName: preprocess_yolo_boxes.py
# @Software: PyCharm

import os
import numpy as np
from PIL import Image
from Preprocess.utils import Data_augmentation

def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


def preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes):
    assert (true_boxes[..., 4]<num_classes).all(), 'class id must be less than num_classes'
    # 一共有三个特征层数
    num_layers = len(anchors)//3 # 9//3
    # 先验框
    # 678为 142,110,  192,243,  459,401
    # 345为 36,75,  76,55,  72,146
    # 012为 12,16,  19,36,  40,28
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]]
    # print('Anchor mask: {}'.format(anchor_mask))

    true_boxes = np.array(true_boxes, dtype='float32')  # N, 100, 5
    input_shape = np.array(input_shape, dtype='int32')  # 160, 180

    # 读出xy轴，读出长宽
    # 中心点(m,n,2)
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2  # 中心点
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]         # 长宽
    # 计算比例
    # print('input_shape[::-1] ', input_shape[::-1])
    true_boxes[..., 0:2] = boxes_xy/input_shape[::-1] # 是对字符串的截取操作
    true_boxes[..., 2:4] = boxes_wh/input_shape[::-1]

    # m张图
    m = true_boxes.shape[0]
    # 得到网格的shape为13,13;26,26;52,52 (406,406) || 5,15;10,30;20,60 (160, 480)
    grid_shapes = [input_shape//{0:32, 1:16, 2:8}[l] for l in range(num_layers)]
    # y_true的格式为(m,13,13,3,85)(m,26,26,3,85)(m,52,52,3,85) || (m,5,15,3,13)(m,10,30,3,13)(m,20,60,3,13)
    # grid_h, grid_w, layer_anchor_number, 5+num_classes
    y_true = [np.zeros((m,grid_shapes[l][0], grid_shapes[l][1], len(anchor_mask[l]), 5+num_classes),dtype='float32') for l in range(num_layers)]
    # for true in y_true:
    #     print('true: ', np.shape(true))
    # [1,9,2]
    anchors = np.expand_dims(anchors, 0)
    anchor_maxes = anchors / 2.
    anchor_mins = -anchor_maxes
    # print('anchor_maxes: \n', anchor_maxes)
    # print('anchor_mins: \n', anchor_mins)

    # 长宽要大于0才有效
    valid_mask = boxes_wh[..., 0]>0  # 426, 100
    # print('anchor_maxes: \n', np.shape(valid_mask))

    for b in range(m):
        # 对每一张图进行处理
        wh = boxes_wh[b, valid_mask[b]] # boxes_wh: (426, 100, 2) 部分wh有效

        if len(wh)==0: continue
        # [n,1,2]
        wh = np.expand_dims(wh, -2)
        box_maxes = wh / 2.
        box_mins = -box_maxes
        # 关于0中心化

        # 计算真实框和哪个先验框最契合
        intersect_mins = np.maximum(box_mins, anchor_mins)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)
        best_anchor = np.argmax(iou, axis=-1)
        # print('best anchor: ', best_anchor)

        for t, n in enumerate(best_anchor):
            for l in range(num_layers):
                if n in anchor_mask[l]:
                    # floor用于向下取整
                    i = np.floor(true_boxes[b,t,0]*grid_shapes[l][1]).astype('int32') # w
                    j = np.floor(true_boxes[b,t,1]*grid_shapes[l][0]).astype('int32') # h
                    # 找到真实框在特征层l中第b副图像对应的位置
                    k = anchor_mask[l].index(n)
                    c = true_boxes[b, t, 4].astype('int32') # 目标类别
                    # 根据 best anchor 中心 设定 框 x_min, y_min, x_max, y_max
                    y_true[l][b, j, i, k, 0:4] = true_boxes[b, t, 0:4]
                    # 根据 best anchor，如果有物体，中心点为类别 1，无物体为 0
                    y_true[l][b, j, i, k, 4] = 1
                    # onehot vector 物体识别
                    y_true[l][b, j, i, k, 5+c] = 1

    return y_true


if __name__ == "__main__":
    annotation_path =  '../Preparation/data_txt'
    annotation_lines = open(os.path.join(annotation_path, 'kitti_ssd_obj_test.txt')).readlines()
    print(len(annotation_lines))

    data_aug = Data_augmentation(input_shape=(160, 480),
                                 max_boxes=100,
                                 jitter=.3,
                                 hue=.1,
                                 sat=1.5,
                                 val=1.5,
                                 visual=False)

    image_data, box_data = [], []
    for annotation_line in annotation_lines:
        line = annotation_line.split()
        image = Image.open(line[0])
        box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])
        aug_image, aug_box = data_aug.main(image, box)
        image_data.append(aug_image)
        box_data.append(aug_box)
    print('image_data: {}, box_data: {}'.format(np.shape(image_data), np.shape(box_data)))

    classes_path = '../Preparation/data_txt/kitti_classes.txt'
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    print(class_names, num_classes)

    anchors_path = '../Preparation/data_txt/kitti_yolov4_anchors.txt'
    yolo_anchors = get_anchors(anchors_path)
    print(yolo_anchors, np.shape(yolo_anchors))

    image_data = np.array(image_data)
    box_data = np.array(box_data)
    yolo_gt = preprocess_true_boxes(box_data, (160, 480), yolo_anchors, num_classes)
    for gt in yolo_gt:
        print('true: ', np.shape(gt))