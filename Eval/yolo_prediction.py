# -*- coding: utf-8 -*-
# @Time    : 2021/10/11 18:32
# @Author  : Zeqi@@
# @FileName: yolo_prediction.py
# @Software: PyCharm

import os
import sys

sys.path.append(os.path.join(os.getcwd(), ".."))

import colorsys
import logging
# Opensource libs
import os
import shutil
import time
import xml.etree.ElementTree as ET

import cv2
import numpy as np
# tensorflow
import tensorflow as tf
from Callbacks.plot_utils import plot_one_box
from Models.tiny_yolov4 import tiny_yolov4
# Callbacks
from Models.yolo_val import yolo_eval
# Model
from Models.yolov4 import yolov4
from PIL import Image
from tensorflow.compat.v1.keras import backend as K
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tensorflow.keras.models import Model
from tqdm import tqdm
# utils
from Utils.utils import get_anchors, get_classes

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger('Prediction - yolov4')
logger.setLevel(logging.DEBUG)


def letterbox_image(image, size, bg_zero):
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    if bg_zero:
        new_image = Image.new('RGB', size, (0,0,0))
    else: 
        new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image

def excute_map_calculation(results_path='input'):
    if os.path.exists(results_path):
        shutil.rmtree(results_path)
        print('Cleaned the existing folder')
    os.makedirs(results_path)
    print('Created a new folder')
    gt_path = os.path.join(results_path, 'ground-truth')
    pre_path = os.path.join(results_path, 'detection-results')
    os.makedirs(gt_path, exist_ok=True)
    os.makedirs(pre_path, exist_ok=True)

class Predictor:
    def __init__(self, **kwargs):
        # load model parameters
        self.add_bn = kwargs.get('add_bn', True)
        self.add_bias = kwargs.get('add_bias', False)
        self.input_shape = kwargs.get('input_shape', (416, 416, 3))
        self.model_path = kwargs.get('model_path', None)

        self.score = kwargs.get('score', 0.5)
        self.iou = kwargs.get('iou', 0.3)
        self.max_boxes = kwargs.get('max_boxes', 100)
        self.bg_zero = kwargs.get('bg_zero', False)
        self.letterbox_image = kwargs.get('letterbox_image', True)

        classes_path = kwargs.get('classes_path', None)
        anchors_path = kwargs.get('anchors_path', None)
        self.class_names = get_classes(classes_path)
        self.anchors = get_anchors(anchors_path)

        # write down prediction results
        self.write_down = kwargs.get('write_down', False)
        self.pre_path = kwargs.get('pre_path', "inputs/detection-results")

        tf.compat.v1.disable_eager_execution() # 很重要
        self.sess = K.get_session()
        self.generate()

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # 计算anchor数量
        self.num_anchors = len(self.anchors)
        self.num_classes = len(self.class_names)

        # 载入模型，如果原来的模型里已经包括了模型结构则直接载入。
        # 否则先构建模型再载入
        # self.inference_model.summary()
        # self.yolo_weights = self.inference_model.get_weights()
        inputs = Input(shape=self.input_shape)
        self.yolo_model = yolov4(num_anchors=self.num_anchors // 3,
                                 num_classes= self.num_classes,
                                 add_bias=self.add_bias,
                                 add_bn=self.add_bn)(inputs)

        self.yolo_model.load_weights(self.model_path, by_name=True, skip_mismatch=True)
        logger.info('{} model, {} anchors, and {} classes loaded.'.format(model_path, self.num_anchors, self.num_classes))

        # 画框设置不同的颜色
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

        # 打乱颜色
        np.random.seed(10101)
        np.random.shuffle(self.colors)
        np.random.seed(None)

        # 预测
        self.input_image_shape = K.placeholder(shape=(2,))
        self.boxes, self.scores, self.classes = yolo_eval(yolo_outputs=self.yolo_model.output,
                                                          anchors=self.anchors,
                                                          num_classes=self.num_classes,
                                                          image_shape=self.input_image_shape,
                                                          max_boxes=self.max_boxes,
                                                          score_threshold=self.score,
                                                          iou_threshold=self.iou, 
                                                          letterbox_image=self.letterbox_image)

        # logging.info('Prediction: boxes: {}, scores: {}, classes: {}'.format(np.shape(self.boxes),
        #                                                                      np.shape(self.scores),
        #                                                                      np.shape(self.classes)))


    def detect_image(self, image, image_id, *args, **kwargs):
        gt_boxes = kwargs.get('gt_boxes', False)
        plot_gt = kwargs.get('plot_gt', False)
        plot_pt = kwargs.get('plot_pt', False)

        if self.write_down:
            self.detect_txtfile = open(os.path.join(self.pre_path, image_id + ".txt"), "w")

        start = time.time()

        # Copy the original image
        raw_image = image.copy()
        raw_image = np.array(raw_image, dtype=np.uint8)

        # 调整图片使其符合输入要求
        new_image_size = (self.input_shape[1], self.input_shape[0])
        boxed_image =letterbox_image(image, new_image_size, self.bg_zero)
        image_data = np.array(boxed_image, dtype='float32')
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        # 预测结果
        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        # logging.info('out boxes: {}, out_scores: {}, out_classes: {}'.format(out_boxes, out_scores, out_classes))

        if gt_boxes and plot_gt:
            for gt_box in gt_boxes:
                gt_name, gt_left, gt_top, gt_right, gt_bottom = gt_box

                plot_one_box(raw_image, [int(gt_left), int(gt_top), int(gt_right), int(gt_bottom)],
                        class_names=self.class_names,
                        label=gt_name,
                        color=(0, 200, 0))

        for i, c in list(enumerate(out_classes)):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            top, left, bottom, right = box
            # top = top - 5
            # left = left - 5
            # bottom = bottom + 5
            # right = right + 5
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

            # 画框框
            if plot_pt:
                plot_one_box(raw_image, [int(left), int(top), int(right), int(bottom)],
                            class_names=self.class_names,
                            label=predicted_class + ', {:.2f}%'.format(np.round(float(score) * 100,2)),
                            color=self.colors[int(c)-1])

            if self.write_down:
                self.detect_txtfile.write("%s %s %s %s %s %s\n" % (
                predicted_class, score, str(int(left)), str(int(top)), str(int(right)), str(int(bottom))))

        end = time.time()
        inference_time = np.round((end - start) * 100, 2)
        fps = int(1 / (end - start))

        # logging.info('Inference time: {} ms, FPS: {}'.format(inference_time, fps))

        if self.write_down:
            self.detect_txtfile.close()

        return raw_image

    def close_session(self):
        self.sess.close()

class get_gt():
    def __init__(self, 
                data_path, 
                gt_path, 
                *args, **kwargs):
        self.gt_path = gt_path
        self.data_path = data_path
        self.image_dir = os.path.join(data_path, 'JPEGImages')
        self.anno_dir = os.path.join(data_path, 'Annotations')

    def __call__(self):
        ground_truth = {}

        for image_id in tqdm(os.listdir(self.image_dir)):
            image_id = image_id.split('.')[0]

            with open(os.path.join(self.gt_path, image_id + ".txt"), "w") as new_f:
                root = ET.parse(os.path.join(self.anno_dir, image_id + ".xml")).getroot()
                ground_truth_per_frame=[]
                for obj in root.findall('object'):
                    obj_name = obj.find('name').text
                    bndbox = obj.find('bndbox')
                    left = bndbox.find('xmin').text
                    top = bndbox.find('ymin').text
                    right = bndbox.find('xmax').text
                    bottom = bndbox.find('ymax').text
                    new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
                    ground_truth_per_frame.append([obj_name, left, top, right, bottom])
                # ground_truth.append(ground_truth_per_frame)
                ground_truth[image_id] = ground_truth_per_frame
        
        return ground_truth


if __name__ == '__main__':
    # clean the results folder
    excute_map_calculation(results_path='input')

    # write down the ground truth
    gt_params = {'data_path': '/Users/babalia/Desktop/Applications/object_detection/yolov5_samples/data/voc/test/VOCdevkit/VOC2007',
                 'gt_path' : 'input/ground-truth',
                }
    gt = get_gt(**gt_params)
    gt_list = gt()

    pt_params = {# model
                'add_bn': True,
                'add_bias': False,
                'input_shape':  (416, 416, 3),
                'model_path': '../Trainer/logs/yolov4_voc_weights_416_fp16/ep190-loss25.458-val_loss13.749.h5',
                
                'score': 0.01,
                'iou': 0.3,
                'max_boxes': 100,
                'letterbox_image': False,
                'bg_zero': False,
                
                "classes_path"       : '../Preparation/data_txt/voc_obj/voc_names.txt',
                "anchors_path"      : '../Preparation/data_txt/voc_obj/yolov4_anchors_416_416.txt',
                
                'write_down': True,
                'pre_path': 'input/detection-results'}

    pt = Predictor(**pt_params)
   
    image_dir = '/Users/babalia/Desktop/Applications/object_detection/yolov5_samples/data/voc/test/VOCdevkit/VOC2007/JPEGImages'


    for image_id in tqdm(os.listdir(image_dir)):
        img_path = os.path.join(image_dir, image_id)
        image = Image.open(img_path)
        image_id = image_id.split('.')[0]

        gt_boxes = gt_list[image_id]
        r_image = pt.detect_image(image, image_id, 
                                  gt_boxes=gt_boxes,
                                  plot_gt=False,
                                  plot_pt=False)

        num_image = np.array(r_image)
        bgr_image = cv2.cvtColor(num_image, cv2.COLOR_RGB2BGR)
        cv2.imshow('demo', bgr_image)
        cv2.waitKey(1000)



