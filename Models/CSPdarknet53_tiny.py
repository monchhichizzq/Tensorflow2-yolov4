# -*- coding: utf-8 -*-
# @Time    : 2021/1/30 20:27
# @Author  : Zeqi@@
# @FileName: CSPdarknet53_tiny.py
# @Software: PyCharm


import logging
import tensorflow as tf
from functools import reduce
from tensorflow.keras import backend as K
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger('Backbone - CSPdarknet53_tiny')
logger.setLevel(logging.DEBUG)



class tiny_cspdarknet53:
    def __init__(self,  **kwargs):
        self.add_bn = kwargs.get('add_bn', True)
        self.use_bias = kwargs.get('add_bias', False)

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

    def route_group(self, input_layer, groups, group_id):
        # 对通道数进行均等分割，我们取第二部分
        convs = tf.split(input_layer, num_or_size_splits=groups, axis=-1)
        return convs[group_id]

    def resblock_body(self, x, num_filters, block):
        '''

                        input
                          |
                DarknetConv2D_BN_Leaky
                          -----------------------
                          |                     |
                     route_group              route
                          |                     |
                DarknetConv2D_BN_Leaky          |
                          |                     |
        -------------------                     |
        |                 |                     |
     route_1    DarknetConv2D_BN_Leaky          |
        |                 |                     |
        -------------Concatenate                |
                          |                     |
            ----DarknetConv2D_BN_Leaky          |
            |             |                     |
          feat       Concatenate-----------------
                          |
                     MaxPooling2D

        :param x:
        :param num_filters:
        :return:
        '''
        # 利用一个3x3卷积进行特征整合
        x = self.DarknetConv2D_BN_Leaky(x, block+'_1', num_filters, (3,3))
        # 引出一个大的残差边route
        route = x

        # 对特征层的通道进行分割，取第二部分作为主干部分。
        x = Lambda(self.route_group,arguments={'groups':2, 'group_id':1})(x)
        # 对主干部分进行3x3卷积
        x = self.DarknetConv2D_BN_Leaky(x, block+'_2', int(num_filters/2), (3,3))
        # 引出一个小的残差边route_1
        route_1 = x
        # 对第主干部分进行3x3卷积
        x = self.DarknetConv2D_BN_Leaky(x, block+'_3', int(num_filters/2), (3,3))
        # 主干部分与残差部分进行相接
        x = Concatenate()([x, route_1])

        # 对相接后的结果进行1x1卷积
        x = self.DarknetConv2D_BN_Leaky(x, block+'_4', num_filters, (1,1))
        feat = x
        x = Concatenate()([route, x])

        # 利用最大池化进行高和宽的压缩
        x = MaxPooling2D(pool_size=[2,2],)(x)

        return x, feat


    def __call__(self, x, *args, **kwargs):

        # 首先利用两次步长为2x2的3x3卷积进行高和宽的压缩
        # 416,416,3 -> 208,208,32 -> 104,104,64
        x = ZeroPadding2D(((1, 0), (1, 0)))(x)
        x = self.DarknetConv2D_BN_Leaky(x, '1', 32, (3, 3), strides=(2, 2))
        x = ZeroPadding2D(((1, 0), (1, 0)))(x)
        x = self.DarknetConv2D_BN_Leaky(x, '2', 64, (3, 3), strides=(2, 2))

        # 104,104,64 -> 52,52,128
        x, _ = self.resblock_body(x, num_filters=64, block='block1')
        # 52,52,128 -> 26,26,256
        x, _ = self.resblock_body(x, num_filters=128, block='block2')
        # 26,26,256 -> x为13,13,512
        #           -> feat1为26,26,256
        x, feat1 = self.resblock_body(x, num_filters=256, block='block3')
        # 13,13,512 -> 13,13,512
        x = self.DarknetConv2D_BN_Leaky(x, '3', 512, (3, 3))

        feat2 = x
        return feat1, feat2


if __name__ == '__main__':
    input = Input(shape=(416, 416, 3))
    feat1, feat2 = tiny_cspdarknet53(add_bias=False, add_bn=True)(input)
    model = Model(inputs=input, outputs=[feat1, feat2], name='Tiny-CSPdarknet53')
    model.summary()

