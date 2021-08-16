# -*- coding: utf-8 -*-
# @Time    : 2020/01/31 4:37
# @Author  : Zeqi@@
# @FileName: train.py
# @Software: PyCharm

# Opensource libs
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
sys.path.append(os.path.join(os.getcwd(), ".."))

import logging
import numpy as np

# Tensorflow
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau, EarlyStopping
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
from tensorflow.keras.mixed_precision import experimental as mixed_precision

# Model
from Models.yolov4 import yolov4
from Models.tiny_yolov4 import tiny_yolov4

# Loss
from Loss.loss import yolo_loss

# Callbacks
from Callbacks.checkpoints import ModelCheckpoint
from Callbacks.CosineDecay import WarmUpCosineDecayScheduler
from Preprocess.data_loader import Yolo_DataGenerator
# from Callbacks.mAP_yolo_Callbacks import VOC2012mAP_Callback

# utils
from Utils.utils import get_classes, get_anchors

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger('Backbone - CSPdarknet53')
logger.setLevel(logging.DEBUG)


config = {'training': True,
          'mix': True,

          # Hyperparameters
          'lr':1e-3, # adam 1e-3
          'batch_size': 8,
          'input_shape': (416, 416, 3),
          'Cosine_scheduler':False,
          'mosaic': True,
          'epochs': 5000,
          'plot': False,
          'label_smoothing':0.1,
          'bg_zero': False,

          # Model
          'tiny-yolov4': False,
          'add_bias': False,
          'add_bn': True,

          # Weights
          'load_weights': False,
          'weights_path' : 'logs/yolov4/mask_recog/ep039-loss14.628-val_loss8.601.h5',

          # checkpoints
          'save_best_only': True,
          'save_weights_only': True,
          'log_dir':'logs/yolov4_voc_weights_416_fp16_adam/',

          # MAP
          'map_plot':True,
          'map_command_line': 'python Callbacks/get_map.py',

          # Path
          'anno_train_txt': '../Preparation/data_txt/voc_obj/data/voc_train.txt',
          'anno_test_txt': '../Preparation/data_txt/voc_obj/data/voc_test.txt',
          'classes_path': '../Preparation/data_txt/voc_obj/voc_names.txt',
          'anchors_path': '../Preparation/data_txt/voc_obj/yolov4_anchors_416_416.txt'
}



train_params = {'train': True,
                'is_plot': config['plot'],
                'mosaic': config['mosaic'],
                'background_zero': config['bg_zero'],

                'batch_size': config['batch_size'],
                'input_shape': config['input_shape'],

                'annotation_txt': config['anno_train_txt'],
                'classes_path': config['classes_path'],
                'anchors_path': config['anchors_path']}

val_params = {'train': False,
              'is_plot': config['plot'],
              'mosaic': config['mosaic'],
              'background_zero': config['bg_zero'],

              'batch_size': config['batch_size'],
              'input_shape': config['input_shape'],

              'annotation_txt': config['anno_test_txt'],
              'classes_path': config['classes_path'],
              'anchors_path': config['anchors_path']}

train_generator = Yolo_DataGenerator(**train_params)
val_generator = Yolo_DataGenerator(**val_params)

num_train = train_generator.num
num_val = val_generator.num


if __name__ == "__main__":
    if config['mix']:
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_policy(policy)

        logger.info('Compute dtype: %s' % policy.compute_dtype)
        logger.info('Variable dtype: %s' % policy.variable_dtype)

    # data_path = config['data_path']
    classes_path = config['classes_path']
    anchors_path = config['anchors_path']

    class_names = get_classes(classes_path)
    anchors = get_anchors(anchors_path)

    num_classes = len(class_names)
    num_anchors = len(anchors)

    mosaic = config['mosaic']
    Cosine_scheduler = config['Cosine_scheduler']
    label_smoothing = config['label_smoothing']

    h, w, c = config['input_shape']
    image_input = Input(shape=(h, w, c))

    if config['tiny-yolov4']:
        logger.info('Create tiny YOLOv4 model with {} anchors and {} classes.'.format(num_anchors, num_classes))
        inputs = Input(shape=config['input_shape'])
        model_body = tiny_yolov4(num_anchors=num_anchors // 3, 
                                 num_classes=num_classes, 
                                 add_bias=config['add_bias'], 
                                 add_bn=config['add_bn'])(inputs)

    else:
        logger.info('Create YOLOv4 model with {} anchors and {} classes.'.format(num_anchors, num_classes))
        inputs = Input(shape=config['input_shape'])
        model_body = yolov4(num_anchors=num_anchors // 3, 
                            num_classes=num_classes, 
                            add_bias=config['add_bias'],
                            add_bn=config['add_bn'])(inputs)

    # model_body.summary()
    if config['load_weights']:
        weights_path = config['weights_path']
        logger.info('Load weights {}.'.format(weights_path))
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)


    # y_true
    if config['tiny-yolov4']:
        # 13,13,3,85
        # 26,26,3,85
        y_true = [Input(shape=(h // {0: 32, 1: 16}[l], w // {0: 32, 1: 16}[l],
                               num_anchors // 3, num_classes + 5)) for l in range(3)]
    else:
        # 13,13,3,85
        # 26,26,3,85
        # 52,52,3,85
        y_true = [Input(shape=(h // {0: 32, 1: 16, 2: 8}[l], w // {0: 32, 1: 16, 2: 8}[l],
                               num_anchors // 3, num_classes + 5)) for l in range(3)]


    print('model_body.output: ', model_body.output)
    #输入为*model_body.input, *y_true
    #输出为model_loss
    loss_input = [*model_body.output, *y_true]
    #  yolo_loss(args, anchors, num_classes, ignore_thresh=.5, label_smoothing=0.1, print_loss=False):
    model_loss = Lambda(yolo_loss, output_shape=(1,), 
                        name='yolo_loss',
                        arguments={'anchors': anchors, 
                                   'num_classes': num_classes, 
                                   'ignore_thresh': 0.5,
                                   'label_smoothing': label_smoothing})(loss_input)
    # 构建了以图片数据和图片标签（y_true）为输入，模型损失（model_loss）为输出（y_pred）的模型 model
    model = Model(inputs=[model_body.input, *y_true], outputs=model_loss)
    logger.info('Inputs {}.'.format(model.input))
    logger.info('Output {}.'.format(model.output))
    #  model.summary()

    # save
    log_dir = config['log_dir']
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 训练参数设置
    # logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + "/ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5",
                                 save_weights_only=config['save_weights_only'],
                                 save_best_only=config['save_best_only'],
                                 period=1)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=30, verbose=1)

    train_callbacks = [checkpoint, reduce_lr]
    val_callbacks = []


    def check_layer_dtype(model):
        for layer in model.layers:
            input_type = layer.input[0].dtype if type(layer.input) is list else layer.input.dtype
            output_type = layer.output[0].dtype if type(layer.output) is list else layer.output.dtype
            logger.info('{} - Input: {} - Output: {}'.format(layer.name, input_type, output_type))


    learning_rate_base = config['lr']
    opt = Adam(learning_rate_base)
    # opt = SGD(learning_rate=learning_rate_base, momentum=0.9)

    # 这纯粹是一种python语法，可以看成一个函数 输入(y_true, y_pred) 返回 y_pred
    # https://zhuanlan.zhihu.com/p/112885878
    # 解释：模型compile时传递的是自定义的loss，而把loss写成一个层融合到model里面后，y_pred就是loss。自定义损失函数规定要以y_true, y_pred为参数。
    model.compile(optimizer=opt, loss={'yolo_loss': lambda y_true, y_pred: y_pred})

    if config['training']:
        # check_layer_dtype(model)

        logger.info('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, config['batch_size']))

        model.fit(train_generator,
                  validation_data=val_generator,
                  epochs=config['epochs'],
                  verbose=2,
                  callbacks=train_callbacks)

    else:
        model.predict(val_generator, verbose=1, callbacks=val_callbacks)


