# -*- coding: utf-8 -*-
# @Time    : 2020/01/31 4:37
# @Author  : Zeqi@@
# @FileName: train.py
# @Software: PyCharm

# Opensource libs
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import logging
import numpy as np

# Tensorflow
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau, EarlyStopping
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
from tensorflow.keras.mixed_precision import experimental as mixed_precision

# Model
from Models.yolov4 import yolov4
from Models.tiny_yolov4 import tiny_yolov4

# Loss
from Loss.yolo_loss import yolo_loss

# Callbacks
from Callbacks.checkpoints import ModelCheckpoint
from Callbacks.CosineDecay import WarmUpCosineDecayScheduler
from Preprocess.data_loader import Kitti_Yolo_DataGenerator
from Callbacks.mAP_yolo_Callbacks import VOC2012mAP_Callback

# utils
from Utils.utils import get_classes, get_anchors


logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger('Backbone - CSPdarknet53')
logger.setLevel(logging.DEBUG)


config = {'training': True,
          'mix': True,

          # Hyperparameters
          'lr':1e-3,
          'batch_size': 8,
          'input_shape': (416, 416, 3),
          'Cosine_scheduler':False,
          'mosaic': True,
          'epochs': 5000,
          'plot':False,
          'label_smoothing':0,

          # Model
          'tiny-yolov4': False,
          'add_bias': True,
          'add_bn': True,

          # Weights
          'load_weights': True,
          'weights_path' : 'logs/yolov4/mask_recog/ep039-loss14.628-val_loss8.601.h5',

          # checkpoints
          'save_best_only': False,
          'save_weights_only': True,
          'log_dir':'logs/yolov4/mask_recog/',

          # MAP
          'map_plot':True,
          'map_command_line': 'python Callbacks/get_map.py',


          # Path
          'data_path': 'E:/mask_detection/face_mask',
          'annotation_path': 'Preparation/data_txt',
          'anno_train_txt': 'face_mask_train.txt',
          'anno_val_txt': 'face_mask_val.txt',
          'classes_path': 'Preparation/data_txt/mask_classes.txt',
          'anchors_path': 'Preparation/data_txt/Mask_yolov4_anchors_416_416.txt',


}


train_params = {'train': True,
                'batch_size': config['batch_size'],
                'input_shape': config['input_shape'],
                'mosaic': config['mosaic'],
                'data_path': config['data_path'],
                'annotation_path': config['annotation_path'],
                'anno_train_txt':config['anno_train_txt'],
                'anno_val_txt': config['anno_val_txt'],
                'classes_path': config['classes_path'],
                'anchors_path': config['anchors_path'],
                'plot': config['plot']}

train_generator = Kitti_Yolo_DataGenerator(**train_params)

val_params = {'train': False,
              'batch_size': config['batch_size'],
              'input_shape': config['input_shape'],
              'mosaic': False,
              'data_path': config['data_path'],
              'annotation_path': config['annotation_path'],
              'anno_train_txt': config['anno_train_txt'],
              'anno_val_txt': config['anno_val_txt'],
              'classes_path': config['classes_path'],
              'anchors_path': config['anchors_path'],
              'plot': False}
              # 'plot': config['plot']}

val_generator = Kitti_Yolo_DataGenerator(**val_params)

num_train = train_generator.num_train
num_val = val_generator.num_val


if __name__ == "__main__":
    if config['mix']:
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_policy(policy)

        logger.info('Compute dtype: %s' % policy.compute_dtype)
        logger.info('Variable dtype: %s' % policy.variable_dtype)

    data_path = config['data_path']
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
        model_body = tiny_yolov4(num_anchors=num_anchors // 3, num_classes=num_classes, add_bias=config['add_bias'], add_bn=config['add_bn'])(inputs)

    else:
        logger.info('Create YOLOv4 model with {} anchors and {} classes.'.format(num_anchors, num_classes))
        inputs = Input(shape=config['input_shape'])
        model_body = yolov4(num_anchors=num_anchors // 3, num_classes=num_classes, add_bias=config['add_bias'],
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

    #输入为*model_body.input, *y_true
    #输出为model_loss
    loss_input = [*model_body.output, *y_true]
    #  yolo_loss(args, anchors, num_classes, ignore_thresh=.5, label_smoothing=0.1, print_loss=False):
    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
                        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5,
                                   'label_smoothing': label_smoothing})(loss_input)

    model = Model([model_body.input, *y_true], model_loss)
    # model = Model(model_body.input, *model_body.output)
    # model = model_body
    model.summary()
    logger.info('Inputs {}.'.format(model.input))
    logger.info('Output {}.'.format(model.output))

    # save
    log_dir = config['log_dir']
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 训练参数设置
    logging = TensorBoard(log_dir=log_dir)
    # checkpoint = ModelCheckpoint(log_dir + "/ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}-mAP{mAP:.2f}.h5",
    #                              save_weights_only=config['save_weights_only'],
    #                              save_best_only=config['save_best_only'],
    #                              period=1)
    checkpoint = ModelCheckpoint(log_dir + "/ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5",
                                 save_weights_only=config['save_weights_only'],
                                 save_best_only=config['save_best_only'],
                                 period=1)

    # early_stopping = EarlyStopping(min_delta=0, patience=50, verbose=1)



    map_config = {'data_path': config['data_path'],
                  'weight_path': config['weights_path'],
                  'classes_path': config['classes_path'],
                  'anchors_path': config['anchors_path'],

                  'add_bias' : config['add_bias'],
                  'add_bn': config['add_bn'],

                  'visual': config['map_plot'],
                  'command_line': config['map_command_line']}

    map_callbacks = VOC2012mAP_Callback(**map_config)


    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20, verbose=1)

    train_callbacks = [map_callbacks, checkpoint, reduce_lr]
    val_callbacks = [map_callbacks]


    def check_layer_dtype(model):
        for layer in model.layers:
            input_type = layer.input[0].dtype if type(layer.input) is list else layer.input.dtype
            output_type = layer.output[0].dtype if type(layer.output) is list else layer.output.dtype
            logger.info('{} - Input: {} - Output: {}'.format(layer.name, input_type, output_type))


    learning_rate_base = config['lr']

    model.compile(optimizer=Adam(learning_rate_base), loss={'yolo_loss': lambda y_true, y_pred: y_pred})

    if config['training']:
        check_layer_dtype(model)

        logger.info('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, config['batch_size']))


        model.fit(train_generator,
                  validation_data=val_generator,
                  epochs=config['epochs'],
                  verbose=2,
                  callbacks=train_callbacks)

    else:
        model.predict(val_generator, verbose=1, callbacks=val_callbacks)


