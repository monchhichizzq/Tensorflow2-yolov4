# -*- coding: utf-8 -*-
# @Time    : 2020/12/31 4:37
# @Author  : Zeqi@@
# @FileName: train.py
# @Software: PyCharm


import os
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau, EarlyStopping
from Model.yolov4 import yolo_body
from Loss.loss import yolo_loss
from Callbacks.checkpoints import ModelCheckpoint
from Callbacks.CosineDecay import WarmUpCosineDecayScheduler
from Preprocess.data_loader import Kitti_Yolo_DataGenerator
# from utils.utils import get_random_data, get_random_data_with_Mosaic, rand, WarmUpCosineDecayScheduler, ModelCheckpoint
from Callbacks.MeanAP_Callbacks import VOC2012mAP_Callback
from Loss.loss import Yolo_loss



# ---------------------------------------------------#
#   获得类和先验框
# ---------------------------------------------------#
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


gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


input_shape = (160, 480, 3)
batch_size = 8


train_params = {'train': True,
                'batch_size': batch_size,
                'input_shape': input_shape,
                'mosaic': True,
                'data_path': 'H:/Applications/Kitti_dataset/kitti_voc/',
                'annotation_path': 'Preparation/data_txt',
                'classes_path': 'Preparation/data_txt/kitti_classes.txt',
                'anchors_path': 'Preparation/data_txt/kitti_yolov4_anchors.txt',
                'plot': False}

train_generator = Kitti_Yolo_DataGenerator(**train_params)

val_params = {'train': False,
                'batch_size': 1,
                'input_shape': input_shape,
                'mosaic': False,
                'data_path': 'H:/Applications/Kitti_dataset/kitti_voc/',
                'annotation_path': 'Preparation/data_txt',
                'classes_path': 'Preparation/data_txt/kitti_classes.txt',
                'anchors_path': 'Preparation/data_txt/kitti_yolov4_anchors.txt',
                'plot': False}

val_generator = Kitti_Yolo_DataGenerator(**val_params)

num_train = train_generator.num_train
num_val = val_generator.num_val


if __name__ == "__main__":
    training = True

    # 标签的位置
    # annotation_path = '2007_train.txt'
    # 获取classes和anchor的位置
    data_path = 'H:/Applications/Kitti_dataset/kitti_voc/'
    classes_path = 'Preparation/data_txt/kitti_classes.txt'
    anchors_path = 'Preparation/data_txt/kitti_yolov4_anchors.txt'
    # ------------------------------------------------------#
    # ------------------------------------------------------#
    # weights_path = 'model_data/yolo4_weight.h5'
    # 获得classes和anchor
    class_names = get_classes(classes_path)
    anchors = get_anchors(anchors_path)
    # 一共有多少类
    num_classes = len(class_names)
    num_anchors = len(anchors)
    # 输入的shape大小
    # 显存比较小可以使用416x416
    # 现存比较大可以使用608x608
    # input_shape = (416, 416)
    mosaic = True
    Cosine_scheduler = False
    label_smoothing = 0

    # 输入的图像为
    h, w, c = input_shape
    image_input = Input(shape=(h, w, c))


    # 创建yolo模型
    print('Create YOLOv4 model with {} anchors and {} classes.'.format(num_anchors, num_classes))
    model_body = yolo_body(image_input, num_anchors // 3, num_classes, use_bias=False)

    # 载入预训练权重
    weights_path = 'logs/ep161-loss21.142-val_loss12.810.h5'
    print('Load weights {}.'.format(weights_path))
    model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)

    # y_true
    # 13,13,3,85
    # 26,26,3,85
    # 52,52,3,85
    y_true = [Input(shape=(h // {0: 32, 1: 16, 2: 8}[l], w // {0: 32, 1: 16, 2: 8}[l],
                           num_anchors // 3, num_classes + 5)) for l in range(3)]

    #输入为*model_body.input, *y_true
    #输出为model_loss
    loss_input = [*model_body.output, *y_true]

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
                        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5,
                                   'label_smoothing': label_smoothing})(loss_input)

    model = Model([model_body.input, *y_true], model_loss)
    # model = Model(model_body.input, *model_body.output)
    # model = model_body
    model.summary()
    print('Input: ', model.input)
    print('Output: ', model.output)

    # 训练后的模型保存的位置
    log_dir = os.path.join("logs")
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    # 训练参数设置
    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + "/ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5",
                                 save_weights_only=True, save_best_only=True, period=1)
    early_stopping = EarlyStopping(min_delta=0, patience=50, verbose=1)
    # map_callbacks = VOC2012mAP_Callback(data_path,
    #                                     visual=False,
    #                                     command_line="python Callbacks/get_map.py")

    # callbacks = [map_callbacks]


    # ------------------------------------------------------#
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   Init_Epoch为起始世代
    #   Freeze_Epoch为冻结训练的世代
    #   Epoch总训练世代
    #   提示OOM或者显存不足请调小Batch_size
    # ------------------------------------------------------#
    # freeze_layers = 302
    # for i in range(freeze_layers): model_body.layers[i].trainable = False
    # print('Freeze the first {} layers of total {} layers.'.format(freeze_layers, len(model_body.layers)))

    # 调整非主干模型first
    if training:
        Init_epoch = 0
        Freeze_epoch = 5000
        # batch_size大小，每次喂入多少数据

        # 最大学习率
        learning_rate_base = 0.5e-4
        if Cosine_scheduler:
            # 预热期
            warmup_epoch = int((Freeze_epoch - Init_epoch) * 0.2)
            # 总共的步长
            total_steps = int((Freeze_epoch - Init_epoch) * num_train / batch_size)
            # 预热步长
            warmup_steps = int(warmup_epoch * num_train / batch_size)
            # 学习率
            reduce_lr = WarmUpCosineDecayScheduler(learning_rate_base=learning_rate_base,
                                                   total_steps=total_steps,
                                                   warmup_learning_rate=1e-4,
                                                   warmup_steps=warmup_steps,
                                                   hold_base_rate_steps=num_train,
                                                   min_learn_rate=1e-6
                                                   )
            model.compile(optimizer=Adam(), loss={'yolo_loss': lambda y_true, y_pred: y_pred})
        else:
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=50, verbose=1)
            # # model.compile(optimizer=Adam(learning_rate_base), loss={'yolo_loss': lambda y_true, y_pred: y_pred})
            # yolo_loss = Yolo_loss(anchors, num_classes, ignore_thresh=.5, label_smoothing=label_smoothing)
            # model.compile(optimizer=Adam(learning_rate_base), loss=yolo_loss.loss)
            model.compile(optimizer=Adam(learning_rate_base), loss={'yolo_loss': lambda y_true, y_pred: y_pred})

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit(train_generator,
                  # steps_per_epoch=max(1, num_train // batch_size),
                  validation_data=val_generator,
                  # validation_steps=max(1, num_val // batch_size),
                  epochs=Freeze_epoch,
                  initial_epoch=Init_epoch,
                  # max_queue_size=1,
                  verbose=2,
                  callbacks=[logging, checkpoint, reduce_lr],
                  )
        model.save_weights(log_dir + 'trained_weights_stage_1.h5')



    # for i in range(freeze_layers): model_body.layers[i].trainable = True

    # # 解冻后训练
    # if True:
    #     Freeze_epoch = 50
    #     Epoch = 100
    #     # batch_size大小，每次喂入多少数据
    #     batch_size = 2
    #
    #     # 最大学习率
    #     learning_rate_base = 1e-4
    #     if Cosine_scheduler:
    #         # 预热期
    #         warmup_epoch = int((Epoch - Freeze_epoch) * 0.2)
    #         # 总共的步长
    #         total_steps = int((Epoch - Freeze_epoch) * num_train / batch_size)
    #         # 预热步长
    #         warmup_steps = int(warmup_epoch * num_train / batch_size)
    #         # 学习率
    #         reduce_lr = WarmUpCosineDecayScheduler(learning_rate_base=learning_rate_base,
    #                                                total_steps=total_steps,
    #                                                warmup_learning_rate=1e-5,
    #                                                warmup_steps=warmup_steps,
    #                                                hold_base_rate_steps=num_train // 2,
    #                                                min_learn_rate=1e-6
    #                                                )
    #         model.compile(optimizer=Adam(), loss={'yolo_loss': lambda y_true, y_pred: y_pred})
    #     else:
    #         reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    #         model.compile(optimizer=Adam(learning_rate_base), loss={'yolo_loss': lambda y_true, y_pred: y_pred})
    #
    #     print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
    #     model.fit_generator(
    #         data_generator(lines[:num_train], batch_size, input_shape, anchors, num_classes, mosaic=mosaic),
    #         steps_per_epoch=max(1, num_train // batch_size),
    #         validation_data=data_generator(lines[num_train:], batch_size, input_shape, anchors, num_classes,
    #                                        mosaic=False),
    #         validation_steps=max(1, num_val // batch_size),
    #         epochs=Epoch,
    #         initial_epoch=Freeze_epoch,
    #         max_queue_size=1,
    #         callbacks=[logging, checkpoint, reduce_lr, early_stopping])
    #     model.save_weights(log_dir + 'last1.h5')
