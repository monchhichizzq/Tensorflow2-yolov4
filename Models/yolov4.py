# -*- coding: utf-8 -*-
# @Time    : 2020/12/31 3:42
# @Author  : Zeqi@@
# @FileName: yolov4.py
# @Software: PyCharm


import numpy as np
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import concatenate
from Models.CSPdarknet53 import darknet_body
from Utils.utils import get_anchors, get_classes

def DarknetConv2D(x,
                  filters,
                  kernel_size,
                  strides=(1, 1),
                  kernel_regularzier=l2(5e-4),
                  use_bias=False):

    padding = 'valid' if strides == (2,2) else 'same'

    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               strides=strides,
               kernel_regularizer=kernel_regularzier,
               padding=padding,
               use_bias=use_bias)(x)
    return x


def DarknetConv2D_BN_Leaky(x,
                          filters,
                          kernel_size,
                          strides=(1, 1),
                          kernel_regularzier=l2(5e-4),
                          use_bias=False):

    padding = 'valid' if strides == (2,2) else 'same'

    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               strides=strides,
               kernel_regularizer=kernel_regularzier,
               padding=padding,
               use_bias=use_bias)(x)

    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    return x

# ---------------------------------------------------#
#   特征层->最后的输出
# ---------------------------------------------------#
def make_five_convs(x, num_filters, use_bias):
    # 五次卷积
    x = DarknetConv2D_BN_Leaky(x, filters=num_filters, kernel_size=(1, 1), use_bias=use_bias)
    x = DarknetConv2D_BN_Leaky(x, filters=num_filters * 2, kernel_size=(3, 3), use_bias=use_bias)
    x = DarknetConv2D_BN_Leaky(x, filters=num_filters, kernel_size=(1, 1), use_bias=use_bias)
    x = DarknetConv2D_BN_Leaky(x, filters=num_filters * 2, kernel_size=(3, 3), use_bias=use_bias)
    x = DarknetConv2D_BN_Leaky(x, filters=num_filters, kernel_size=(1, 1), use_bias=use_bias)
    return x


# ---------------------------------------------------#
#   特征层->最后的输出
# ---------------------------------------------------#
def yolo_body(inputs, num_anchors, num_classes, use_bias):
    # 生成darknet53的主干模型
    feat1, feat2, feat3 = darknet_body(inputs)

    # 第一个特征层
    # y1=(batch_size,13,13,3,85)
    P5 = DarknetConv2D_BN_Leaky(feat3, filters=512, kernel_size=(1, 1), use_bias=use_bias)
    P5 = DarknetConv2D_BN_Leaky(P5, filters=1024, kernel_size=(3, 3), use_bias=use_bias)
    P5 = DarknetConv2D_BN_Leaky(P5, filters=512, kernel_size=(1, 1), use_bias=use_bias)
    # 使用了SPP结构，即不同尺度的最大池化后堆叠。
    maxpool1 = MaxPooling2D(pool_size=(13, 13), strides=(1, 1), padding='same')(P5)
    maxpool2 = MaxPooling2D(pool_size=(9, 9), strides=(1, 1), padding='same')(P5)
    maxpool3 = MaxPooling2D(pool_size=(5, 5), strides=(1, 1), padding='same')(P5)
    P5 = Concatenate()([maxpool1, maxpool2, maxpool3, P5])
    P5 = DarknetConv2D_BN_Leaky(P5, filters=512, kernel_size=(1, 1), use_bias=use_bias)
    P5 = DarknetConv2D_BN_Leaky(P5, filters=1024, kernel_size=(3, 3), use_bias=use_bias)
    P5 = DarknetConv2D_BN_Leaky(P5, filters=512, kernel_size=(1, 1), use_bias=use_bias)

    P5_conv= DarknetConv2D_BN_Leaky(P5, filters=256, kernel_size=(1, 1), use_bias=use_bias)
    P5_upsample = UpSampling2D(2)(P5_conv)

    P4_conv = DarknetConv2D_BN_Leaky(feat2, filters=256, kernel_size=(1, 1), use_bias=use_bias)
    P4_concat = Concatenate()([P4_conv, P5_upsample])
    P4 = make_five_convs(P4_concat, 256, use_bias)

    P4_conv = DarknetConv2D_BN_Leaky(P4, filters=128, kernel_size=(1, 1), use_bias=use_bias)
    P4_upsample = UpSampling2D(2)(P4_conv)


    P3 = DarknetConv2D_BN_Leaky(feat1, filters=128, kernel_size=(1, 1), use_bias=use_bias)
    P3 = Concatenate()([P3, P4_upsample])
    P3 = make_five_convs(P3, 128, use_bias)


    P3_output = DarknetConv2D_BN_Leaky(P3, filters=256, kernel_size=(3, 3), use_bias=use_bias)
    P3_output = DarknetConv2D(P3_output, filters=num_anchors * (num_classes + 5), kernel_size=(1, 1), use_bias=use_bias)


    # 38x38 output
    P3_downsample = ZeroPadding2D(((1, 0), (1, 0)))(P3)
    P3_downsample = DarknetConv2D_BN_Leaky(P3_downsample, filters=256, kernel_size=(3, 3), strides=(2, 2), use_bias=use_bias)
    P4 = Concatenate()([P3_downsample, P4])
    P4 = make_five_convs(P4, 256, use_bias)

    P4_output = DarknetConv2D_BN_Leaky(P4, filters=512, kernel_size=(3, 3), use_bias=use_bias)
    P4_output = DarknetConv2D(P4_output, filters=num_anchors * (num_classes + 5), kernel_size=(1, 1), use_bias=use_bias)

    # 19x19 output
    P4_downsample = ZeroPadding2D(((1, 0), (1, 0)))(P4)
    P4_downsample = DarknetConv2D_BN_Leaky(P4_downsample, filters=512, kernel_size=(3, 3), strides=(2, 2), use_bias=use_bias)
    P5 = Concatenate()([P4_downsample, P5])
    P5 = make_five_convs(P5, 512, use_bias)

    P5_output = DarknetConv2D_BN_Leaky(P5, filters=1024, kernel_size=(3, 3), use_bias=use_bias)
    P5_output = DarknetConv2D(P5_output, filters=num_anchors * (num_classes + 5), kernel_size=(1, 1), use_bias=use_bias)

    prediction = [P5_output, P4_output, P3_output]

    return Model(inputs, prediction)

if __name__ == '__main__':
    anchors_path = '../Preparation/data_txt/kitti_yolov4_anchors.txt'
    class_path = '../Preparation/data_txt/kitti_classes.txt'
    yolo_anchors = get_anchors(anchors_path)
    print('Yolo anchors: \n {} \n Anchor shape: {} '.format(yolo_anchors, np.shape(yolo_anchors)))
    num_anchors = len(yolo_anchors)
    class_names = get_classes(class_path)
    num_classes = len(class_names)

    inputs = Input(shape=(416, 416, 3))
    model = yolo_body(inputs, num_anchors//3, num_classes, use_bias=False)
    model.summary()
    model.save('Yolov4body.h5')