# -*- coding: utf-8 -*-
# @Time    : 2020/12/29 22:54
# @Author  : Zeqi@@
# @FileName: CSPdarknet53.py.py
# @Software: PyCharm

import logging
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
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger('Backbone - CSPdarknet53')
logger.setLevel(logging.DEBUG)


class Mish(Layer):
    def __init__(self, **kwargs):
        super(Mish, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs):
        return inputs * K.tanh(K.softplus(inputs))

    def get_config(self):
        config = super(Mish, self).get_config()
        return config

    def compute_output_shape(self, input_shape):
        return input_shape



def DarknetConv2D_BN_Mish(x,
                          filters,
                          kernel_size,
                          strides=(1, 1),
                          kernel_regularzier=l2(5e-4),
                          use_bias=False):

    padding = 'valid' if strides==(2,2) else 'same'

    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               strides=strides,
               kernel_regularizer=kernel_regularzier,
               padding=padding,
               use_bias=use_bias)(x)

    x = BatchNormalization()(x)
    x = Mish()(x)
    return x


'''
    CSPdarknet的结构块
    存在一个大残差边
    这个大残差边绕过了很多的残差结构
'''

def resblock_body(x, num_filters, num_blocks, use_bias=False, all_narrow=True):
    # 进行长和宽的压缩
    preconv1 = ZeroPadding2D(((1,0),(1,0)))(x)
    preconv1 = DarknetConv2D_BN_Mish(preconv1, num_filters, (3,3), strides=(2,2), use_bias=use_bias)

    # 生成一个大的残差边
    shortconv = DarknetConv2D_BN_Mish(preconv1, num_filters//2 if all_narrow else num_filters, (1,1), use_bias=use_bias)

    # 主干部分的卷积
    mainconv = DarknetConv2D_BN_Mish(preconv1, num_filters//2 if all_narrow else num_filters, (1,1),  use_bias=use_bias)

    # 1x1卷积对通道数进行整合->3x3卷积提取特征，使用残差结构
    for i in range(num_blocks):
        x = DarknetConv2D_BN_Mish(mainconv, num_filters // 2, (1,1), use_bias=use_bias)
        x = DarknetConv2D_BN_Mish(x, num_filters // 2 if all_narrow else num_filters, (3, 3), use_bias=use_bias)
        mainconv = Add()([mainconv, x])

    # 1x1卷积后和残差边堆叠
    postconv = DarknetConv2D_BN_Mish(mainconv, num_filters//2 if all_narrow else num_filters, (1,1), use_bias=use_bias)
    route = Concatenate()([postconv, shortconv])

    # 最后对通道数进行整合
    return DarknetConv2D_BN_Mish(route, num_filters, (1,1), use_bias=use_bias)

'''
    darknet53 
'''
def darknet_body(x, use_bias=False):
    logger.info('CSPdarknet53')
    x = DarknetConv2D_BN_Mish(x, filters= 32, kernel_size=(3,3), use_bias=use_bias)
    x = resblock_body(x, num_filters=64, num_blocks=1, all_narrow=False) # x, num_filters, num_blocks, all_narrow=True
    x = resblock_body(x, num_filters=128, num_blocks=2)
    x = resblock_body(x, num_filters=256, num_blocks=8)
    feat1 = x
    x = resblock_body(x, num_filters=512, num_blocks=8)
    feat2 = x
    x = resblock_body(x, num_filters=1024, num_blocks=4)
    feat3 = x
    return feat1,feat2,feat3


if __name__ == '__main__':
    input = Input(shape=(416, 416, 3))
    feat1, feat2, feat3 = darknet_body(input)
    model = Model(inputs=input, outputs=[feat1, feat2, feat3], name='CSPdarknet53')
    model.summary()