# -*- coding: utf-8 -*-
# @Time    : 2020/12/31 3:42
# @Author  : Zeqi@@
# @FileName: yolov4.py
# @Software: PyCharm

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import logging
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
from Models.CSPdarknet53 import cspdarknet53
from Utils.utils import get_anchors, get_classes


logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger('Model - Tiny_YoloV4')
logger.setLevel(logging.DEBUG)


class yolov4:
    def __init__(self, num_anchors, num_classes, **kwargs):
        self.add_bn = kwargs.get('add_bn', True)
        self.use_bias = kwargs.get('add_bias', False)
        self.num_anchors = num_anchors
        self.num_classes = num_classes

    def DarknetConv2D(self,
                      x,
                      name,
                      filters,
                      kernel_size,
                      strides=(1, 1),
                      kernel_regularzier=l2(5e-4)):

        padding = 'valid' if strides == (2, 2) else 'same'

        x = Conv2D(filters=filters,
                   kernel_size=kernel_size,
                   strides=strides,
                   kernel_regularizer=kernel_regularzier,
                   padding=padding,
                   use_bias=self.use_bias,
                   kernel_initializer='he_normal',
                   bias_initializer='zeros',
                   name='conv2d_' + name)(x)
        return x

    def DarknetConv2D_BN_Leaky(self,
                               x,
                               name,
                               filters,
                               kernel_size,
                               strides=(1, 1),
                               kernel_regularzier=l2(5e-4)):

        padding = 'valid' if strides == (2, 2) else 'same'

        x = Conv2D(filters=filters,
                   kernel_size=kernel_size,
                   strides=strides,
                   kernel_regularizer=kernel_regularzier,
                   padding=padding,
                   use_bias=self.use_bias,
                   kernel_initializer='he_normal',
                   bias_initializer='zeros',
                   name='conv2d_' + name)(x)

        if self.add_bn:
            x = BatchNormalization(name='conv2d_' + name + '_bn')(x)

        x = LeakyReLU(alpha=0.1, name='conv2d_' + name + '_leaky')(x)
        return x

    def make_five_convs(self, x, num_filters, name):
        x = self.DarknetConv2D_BN_Leaky(x, name+'_1', filters=num_filters, kernel_size=(1, 1))
        x = self.DarknetConv2D_BN_Leaky(x, name+'_2', filters=num_filters * 2, kernel_size=(3, 3))
        x = self.DarknetConv2D_BN_Leaky(x, name+'_3', filters=num_filters, kernel_size=(1, 1))
        x = self.DarknetConv2D_BN_Leaky(x, name+'_4', filters=num_filters * 2, kernel_size=(3, 3))
        x = self.DarknetConv2D_BN_Leaky(x, name+'_5', filters=num_filters, kernel_size=(1, 1))
        return x

    def __call__(self, inputs, *args, **kwargs):
        feat1, feat2, feat3 = cspdarknet53(add_bn=True, add_bias=False)(inputs)

        # 第一个特征层
        # y1=(batch_size,13,13,3,85)
        P5 = self.DarknetConv2D_BN_Leaky(feat3, '2', filters=512, kernel_size=(1, 1))
        P5 = self.DarknetConv2D_BN_Leaky(P5, '3', filters=1024, kernel_size=(3, 3))
        P5 = self.DarknetConv2D_BN_Leaky(P5, '4', filters=512, kernel_size=(1, 1))

        # 使用了SPP结构，即不同尺度的最大池化后堆叠
        maxpool1 = MaxPooling2D(pool_size=(13, 13), strides=(1, 1), padding='same')(P5)
        maxpool2 = MaxPooling2D(pool_size=(9, 9), strides=(1, 1), padding='same')(P5)
        maxpool3 = MaxPooling2D(pool_size=(5, 5), strides=(1, 1), padding='same')(P5)
        P5 = Concatenate()([maxpool1, maxpool2, maxpool3, P5])
        P5 = self.DarknetConv2D_BN_Leaky(P5, '5', filters=512, kernel_size=(1, 1))
        P5 = self.DarknetConv2D_BN_Leaky(P5, '6', filters=1024, kernel_size=(3, 3))
        P5 = self.DarknetConv2D_BN_Leaky(P5, '7', filters=512, kernel_size=(1, 1))

        P5_conv = self.DarknetConv2D_BN_Leaky(P5, '8', filters=256, kernel_size=(1, 1))
        P5_upsample = UpSampling2D(2)(P5_conv)

        P4_conv = self.DarknetConv2D_BN_Leaky(feat2, '9', filters=256, kernel_size=(1, 1))
        P4_concat = Concatenate()([P4_conv, P5_upsample])
        P4 = self.make_five_convs(P4_concat, 256, 'module_1')

        P4_conv = self.DarknetConv2D_BN_Leaky(P4, '10', filters=128, kernel_size=(1, 1))
        P4_upsample = UpSampling2D(2)(P4_conv)

        P3 = self.DarknetConv2D_BN_Leaky(feat1, '11', filters=128, kernel_size=(1, 1))
        P3 = Concatenate()([P3, P4_upsample])
        P3 = self.make_five_convs(P3, 128, 'module_2')

        P3_output = self.DarknetConv2D_BN_Leaky(P3, '12', filters=256, kernel_size=(3, 3))
        P3_output = self.DarknetConv2D(P3_output, '13', filters=num_anchors * (num_classes + 5), kernel_size=(1, 1))

        # 38x38 output
        P3_downsample = ZeroPadding2D(((1, 0), (1, 0)))(P3)
        P3_downsample = self.DarknetConv2D_BN_Leaky(P3_downsample, '14', filters=256, kernel_size=(3, 3), strides=(2, 2))
        P4 = Concatenate()([P3_downsample, P4])
        P4 = self.make_five_convs(P4, 256, 'module_3')

        P4_output = self.DarknetConv2D_BN_Leaky(P4, '15', filters=512, kernel_size=(3, 3))
        P4_output = self.DarknetConv2D(P4_output, '16', filters=num_anchors * (num_classes + 5), kernel_size=(1, 1))

        # 19x19 output
        P4_downsample = ZeroPadding2D(((1, 0), (1, 0)))(P4)
        P4_downsample = self.DarknetConv2D_BN_Leaky(P4_downsample, '17', filters=512, kernel_size=(3, 3), strides=(2, 2))
        P5 = Concatenate()([P4_downsample, P5])
        P5 = self.make_five_convs(P5, 512, 'module_4')

        P5_output = self.DarknetConv2D_BN_Leaky(P5, '18', filters=1024, kernel_size=(3, 3))
        P5_output = self.DarknetConv2D(P5_output, '19', filters=num_anchors * (num_classes + 5), kernel_size=(1, 1))

        prediction = [P5_output, P4_output, P3_output]

        return Model(inputs, prediction)




if __name__ == '__main__':
    anchors_path = '../Preparation/data_txt/BDD100K_yolov4_anchors_416_416.txt'
    class_path = '../Preparation/data_txt/bdd_classes18.txt'
    yolo_anchors = get_anchors(anchors_path)
    logger.info('Yolo anchors: \n {} \n Anchor shape: {} '.format(yolo_anchors, np.shape(yolo_anchors)))
    num_anchors = len(yolo_anchors)
    logger.info('Number of anchors: {} '.format(num_anchors))
    class_names = get_classes(class_path)
    num_classes = len(class_names)
    logger.info('Class number: {} '.format(num_classes))
    logger.info('Class names: \n {} '.format(class_names))

    inputs = Input(shape=(416, 416, 3))
    model = yolov4(num_anchors= num_anchors//3, num_classes=num_classes, add_bias=False, add_bn=True)(inputs)
    model.summary()
    model.save('Yolov4body.h5')