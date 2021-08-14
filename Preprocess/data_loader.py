# -*- coding: utf-8 -*-
# @Time    : 2020/12/28 21:02
# @Author  : Zeqi@@
# @FileName: data_loader.py
# @Software: PyCharm

import os
import sys
sys.path.append(os.path.join(os.getcwd(), ".."))

import cv2
import numpy as np
from tensorflow.keras.utils import Sequence
from Preprocess.utils import Data_augmentation
from Preprocess.mosaic_utils import Data_augmentation_with_Mosaic
from Preprocess.preprocess_yolo_boxes import preprocess_true_boxes, get_anchors, get_classes

# import pandas as pd
# pd.set_option('display.max_columns', 1000)
# pd.set_option('display.width', 1000)
# pd.set_option('display.max_colwidth', 1000)

def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a


class Yolo_DataGenerator(Sequence):
    def __init__(self, train, batch_size, **kwargs):
        # mode selection
        self.train_mode = train
        print('\033[1;34mTrain:\033[0m \033[1;31m{}\033[0m'.format(self.train_mode))
        self.plot = kwargs.get('is_plot', False)
        self.mosaic = kwargs.get('mosaic', True)
        self.shuffle = True if train else False

        # basic parameters
        self.bg_zero = kwargs.get('background_zero', True)
        self.batch_size = batch_size
        self.model_image_size = kwargs.get('input_shape', (160, 480, 3))
        self.input_shape = (self.model_image_size[0], self.model_image_size[1])

        # annotation file path
        self.anno_txt = kwargs.get('annotation_txt', 
                        'Preparation/data_txt/voc_obj/data/voc_train.txt')
        # class name path
        classes_path = kwargs.get('classes_path', 
                        'Preparation/data_txt/voc_obj/voc_names.txt')
        class_names = get_classes(classes_path)
        self.num_classes = len(class_names)
        print('\033[1;34mClass names:\033[0m \n {}'.format(class_names))
        print('\033[1;34mNumber of class:\033[0m {}'.format(self.num_classes))

        # anchors path
        anchors_path = kwargs.get('anchors_path', 
                        'Preparation/data_txt/voc_obj/yolov4_anchors_608_608.txt')
        self.yolo_anchors = get_anchors(anchors_path)
        print('\033[1;34mYolo anchors:\033[0m \n {} \n\033[1;34mAnchor shape:\033[0m {} '.format(self.yolo_anchors, 
                                                                np.shape(self.yolo_anchors)))

        # Read dataset
        self.annotation_lines = open(self.anno_txt).readlines()
        self.num = len(self.annotation_lines)
        print('\033[1;34mNum of training {} samples: {}\033[0m'.format(self.train_mode, self.num))

        # shuffle dataset
        self.on_epoch_end()

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.num)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.num/self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        mosaic_execution = np.random.rand() < 0.5 # 改动

        if self.mosaic and mosaic_execution and self.train_mode:

            image_batch, box_batch = [], []

            for k in indexes:
                # print(k, k+4)
                # in order to get a mosaic image, 4 normal images are needed
                if k+4 > self.num:
                    four_annotation_lines = self.annotation_lines[self.num-4:self.num]    
                else:
                    four_annotation_lines = self.annotation_lines[k:k + 4]
                # 准备好 4 张图 
                mosaic_aug = Data_augmentation_with_Mosaic(four_annotation_lines, 
                                                            input_shape=self.input_shape, 
                                                            visual=False)
                image, box = mosaic_aug.main(background_zero=self.bg_zero)
                image_batch.append(image)
                box_batch.append(box)

        else:
            # Find list of IDs
            annotation_lines_batch = [self.annotation_lines[k] for k in indexes]

            data_aug = Data_augmentation(input_shape=self.input_shape, 
                                         visual=False,
                                         background_zero=self.bg_zero)

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

        
        if self.plot:
            new_image = image.copy()
            new_box = box.copy()
            # print(new_image.shape, np.max(new_image), np.min(new_image))
            new_image = np.array(new_image * 255., dtype=np.uint8)
            # print(new_image.shape, np.max(new_image), np.min(new_image))

            for box in new_box:
                box = [int(b) for b in box]
                cv2.rectangle(new_image, (box[0], box[1]), (box[2], box[3]), color=(255, 255, 255), thickness=1)

            new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
            cv2.imshow('Image', new_image)
            cv2.waitKey(1000)


        # Image batch: 0~1 (bs, 608, 608, 3)
        # label batch: (bs, 100, 5)
        image_batch, box_batch = np.array(image_batch), np.array(box_batch)
        # print('image batch: {}, min: {}, max: {}'.format(np.shape(image_batch), np.min(image_batch), np.max(image_batch)))
        # print('box_batch: {}'.format(np.shape(box_batch)))
        # image_batch = preprocess_input(image_batch) # -127， 127

        label_batch = preprocess_true_boxes(box_batch, self.input_shape, self.yolo_anchors, self.num_classes)

        # [image_data, *y_true], np.zeros(batch_size)

        #return [image_batch, *label_batch]
        return [image_batch, *label_batch], np.zeros(self.batch_size)


if __name__ == '__main__':

    train_params = {'train': True,
                    'is_plot': True,
                    'mosaic': True,

                    'batch_size': 8,
                    'input_shape': (608, 608, 3),
       
                    'annotation_txt': '../Preparation/data_txt/voc_obj/data/voc_train.txt',
                    'classes_path': '../Preparation/data_txt/voc_obj/voc_names.txt',
                    'anchors_path': '../Preparation/data_txt/voc_obj/yolov4_anchors_608_608.txt'
                    }

    train_generator = Yolo_DataGenerator(**train_params)
    _generator = iter(train_generator)
    for i in range(60000):
        image_batch, label_batch = next(_generator)
        # print('Image batch shape: ', np.shape(image_batch))
        # for label_ in label_batch:
        #     print(np.shape(label_))
        # print('')
    
    val_params = {'train': False,
                'is_plot': True,
                'mosaic': False,

                'batch_size': 8,
                'input_shape': (608, 608, 3),
    
                'annotation_txt': '../Preparation/data_txt/voc_obj/data/voc_test.txt',
                'classes_path': '../Preparation/data_txt/voc_obj/voc_names.txt',
                'anchors_path': '../Preparation/data_txt/voc_obj/yolov4_anchors_608_608.txt'
                }

    val_generator = Yolo_DataGenerator(**val_params)
    _generator = iter(val_generator)
    for i in range(60000):
        image_batch, label_batch = next(_generator)

