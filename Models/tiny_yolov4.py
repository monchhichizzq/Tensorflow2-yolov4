# -*- coding: utf-8 -*-
# @Time    : 2021/1/30 20:52
# @Author  : Zeqi@@
# @FileName: tiny_yolov4.py
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
from Models.CSPdarknet53_tiny import tiny_cspdarknet53
from Utils.utils import get_anchors, get_classes

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger('Model - Tiny_YoloV4')
logger.setLevel(logging.DEBUG)


class tiny_yolov4:
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

    def __call__(self, inputs, *args, **kwargs):

        '''
            feat1的shape为26, 26, 256
            feat2的shape为13, 13, 512
        :param args:
        :param kwargs:
        :return:
        '''

        tiny_darknet = tiny_cspdarknet53(add_bias=self.use_bias, add_bn=self.add_bn)
        feat1, feat2 = tiny_cspdarknet53(add_bias=self.use_bias, add_bn=self.add_bn)(inputs)

        # 13,13,512 -> 13,13,256
        P5 = tiny_darknet.DarknetConv2D_BN_Leaky(feat2, '4', 256, (1, 1))
        # 13,13,256 -> 13,13,512 -> 13,13,255
        P5_output = tiny_darknet.DarknetConv2D_BN_Leaky(P5, '5', 512, (3, 3))
        P5_output = self.DarknetConv2D(P5_output, '6', num_anchors * (num_classes + 5), (1, 1))

        # 13,13,256 -> 13,13,128 -> 26,26,128
        P5 = tiny_darknet.DarknetConv2D_BN_Leaky(P5, '7', 128, (1, 1))
        P5_upsample = UpSampling2D(2)(P5)

        # 26,26,256 + 26,26,128 -> 26,26,384
        P4 = Concatenate()([feat1, P5_upsample])

        # 26,26,384 -> 26,26,256 -> 26,26,255
        P4_output = tiny_darknet.DarknetConv2D_BN_Leaky(P4, '8', 256, (3, 3))
        P4_output = self.DarknetConv2D(P4_output, '9', num_anchors * (num_classes + 5), (1, 1))

        return Model(inputs, [P5_output, P4_output])



if __name__ == '__main__':
    anchors_path = '../Preparation/data_txt/BDD100K_tiny_yolov4_anchors_416_416.txt'
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
    model = tiny_yolov4(num_anchors= num_anchors//3, num_classes=num_classes, add_bias=False, add_bn=True)(inputs)
    model.summary()
    model.save('TinyYolov4body.h5')