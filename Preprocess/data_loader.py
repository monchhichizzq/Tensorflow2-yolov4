# -*- coding: utf-8 -*-
# @Time    : 2020/12/28 21:02
# @Author  : Zeqi@@
# @FileName: data_loader.py
# @Software: PyCharm


import sys
sys.path.append('../../Tensorflow_YoloV4_BDD100K')

import os
import numpy as np
from tensorflow.keras.utils import Sequence
from Preprocess.utils import Data_augmentation
from Preprocess.mosaic_utils import Data_augmentation_with_Mosaic
from Preprocess.preprocess_yolo_boxes import preprocess_true_boxes, get_anchors, get_classes

import pandas as pd
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)

def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a


class Kitti_Yolo_DataGenerator(Sequence):
    def __init__(self, train, batch_size, plot = False, **kwargs):
        self.train_mode = train

        self.plot = plot
        self.batch_size = batch_size

        self.data_path = kwargs.get('data_path', 'H:/Applications/Kitti_dataset/kitti_voc/')
        self.annotation_path = kwargs.get('annotation_path', 'Preparation/data_txt')
        self.anno_train_txt = kwargs.get('anno_train_txt', 'bdd100k_obj_train.txt')
        self.anno_val_txt = kwargs.get('anno_val_txt', 'bdd100k_obj_train.txt')

        self.model_image_size = kwargs.get('input_shape', (160, 480, 3))
        self.input_shape = (self.model_image_size[0], self.model_image_size[1])

        # self.consecutive_frames = kwargs.get('consecutive_frames', False)

        self.mosaic = kwargs.get('mosaic', True)

        classes_path = kwargs.get('classes_path', 'Preparation/data_txt/kitti_classes.txt')
        class_names = get_classes(classes_path)
        self.num_classes = len(class_names)
        print('Class names: \n {} \n Number of class: {} '.format(class_names, self.num_classes))

        anchors_path = kwargs.get('anchors_path', 'Preparation/data_txt/kitti_yolov4_anchors.txt')
        self.yolo_anchors = get_anchors(anchors_path)
        print('Yolo anchors: \n {} \n Anchor shape: {} '.format( self.yolo_anchors , np.shape( self.yolo_anchors )))

        if train:
            self.shuffle = True
            # self.image_ids = open(os.path.join(self.data_path, 'ImageSets/Main/trainval.txt')).read().strip().split()
            self.annotation_lines = open(os.path.join(self.annotation_path, self.anno_train_txt)).readlines()
            self.num_train = len(self.annotation_lines)
            print('Num train: ', len(self.annotation_lines))
        else:
            self.shuffle = False
            # self.image_ids = open(os.path.join(self.data_path, 'ImageSets/Main/test.txt')).read().strip().split()
            self.annotation_lines = open(os.path.join(self.annotation_path, self.anno_val_txt)).readlines()
            self.num_val = len(self.annotation_lines)
            print('Num validation: ', len(self.annotation_lines))

        self.on_epoch_end()

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.annotation_lines))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.annotation_lines) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        mosaic_execution = np.random.rand() < 1.5 # 改动

        # print(self.mosaic, mosaic_execution, not self.train_mode)

        if self.mosaic and mosaic_execution and self.train_mode:

            image_batch, box_batch = [], []

            for k in indexes:
                # print(k, k+4)
                if k+4 > self.num_train:
                    four_annotation_lines = self.annotation_lines[self.num_train-4:self.num_train]    # in order to get a mosaic image, 4 normal images are needed
                else:
                    four_annotation_lines = self.annotation_lines[k:k + 4]
                mosaic_aug = Data_augmentation_with_Mosaic(four_annotation_lines, input_shape=self.input_shape, visual=self.plot)

                image_data, box_data = mosaic_aug.main()
                image_batch.append(image_data)
                box_batch.append(box_data)

            # print(index, np.shape(image_batch), np.shape(box_batch), 'mosaic')
        else:

            # Find list of IDs
            annotation_lines_batch = [self.annotation_lines[k] for k in indexes]

            data_aug = Data_augmentation(input_shape=self.input_shape, visual=self.plot)

            # Generate data
            image_batch, box_batch = [], []
            if self.train_mode:
                for annotation_line in annotation_lines_batch:
                    image, box = data_aug.main_train(annotation_line)
                    image_batch.append(image)
                    box_batch.append(box)
                # print(index, np.shape(image_batch), np.shape(box_batch), 'normal')
            else:
                for annotation_line in annotation_lines_batch:
                    image, box = data_aug.main_val(annotation_line)
                    image_batch.append(image)
                    box_batch.append(box)

        # Image batch: 0~1 (160, 480, 3)
        # label batch: (100, 5)
        image_batch, box_batch = np.array(image_batch), np.array(box_batch)
        # image_batch = preprocess_input(image_batch) # -127， 127

        label_batch = preprocess_true_boxes(box_batch, self.input_shape, self.yolo_anchors, self.num_classes)

        # [image_data, *y_true], np.zeros(batch_size)

        #return [image_batch, *label_batch]
        return [image_batch, *label_batch], np.zeros(self.batch_size)


if __name__ == '__main__':

    train_params = {'train': True,
                    'batch_size': 8,
                    'input_shape': (416, 416, 3),
                    'mosaic': False,
                    'data_path': 'D:/BDD100K/',
                    'annotation_path': '../Preparation/data_txt',
                    'classes_path': '../Preparation/data_txt/bdd_classes18.txt',
                    'anchors_path': '../Preparation/data_txt/BDD100K_yolov4_anchors_416_416.txt',
                    'plot': True}

    train_generator = Kitti_Yolo_DataGenerator(**train_params)
    _generator = iter(train_generator)
    for i in range(60000):
        image_batch, label_batch = next(_generator)
        # print(image_batch.shape)
        # for label_ in label_batch:
        #     print(np.shape(label_))
        # print('')
