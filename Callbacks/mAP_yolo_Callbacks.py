# -*- coding: utf-8 -*-
# @Time    : 2021/1/31 18:32
# @Author  : Zeqi@@
# @FileName: mAP_yolo_Callbacks.py
# @Software: PyCharm


# Opensource libs
import os
import cv2
import shutil
import time
import logging
import colorsys
import numpy as np
from tqdm import tqdm
import xml.etree.ElementTree as ET
from PIL import Image, ImageFont, ImageDraw

# Model
from Models.yolov4 import yolov4
from Models.tiny_yolov4 import tiny_yolov4

# Loss
from Loss.loss import yolo_loss

# utils
from Utils.utils import get_classes, get_anchors

# Callbacks
from Callbacks.yolo_eval import yolo_eval

# tensorflow
import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda
from tensorflow.compat.v1.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.mixed_precision import experimental as mixed_precision


logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger('Prediction - yolov4')
logger.setLevel(logging.DEBUG)


class Prediction:
    def __init__(self, **kwargs):
        self.yolo_weights = kwargs.get('weights', None)
        self.write_down = kwargs.get('write_down', True)
        self.load_weights = kwargs.get('load_pretrained_weights', None)

        self.add_bn = kwargs.get('add_bn', True)
        self.add_bias = kwargs.get('add_bias', False)
        self.input_shape = kwargs.get('input_shape', (416, 416, 3))

        self.score = kwargs.get('score', 0.5)
        self.iou = kwargs.get('iou', 0.3)
        self.max_boxes = kwargs.get('max_boxes', 100)

        self.model_path = kwargs.get('weight_path', None)
        classes_path = kwargs.get('classes_path', None)
        anchors_path = kwargs.get('anchors_path', None)
        self.class_names = get_classes(classes_path)
        self.anchors = get_anchors(anchors_path)

        self.pre_path = kwargs.get('pre_path', "input/detection-results")

        tf.compat.v1.disable_eager_execution() # 很重要
        self.sess = K.get_session()
        self.generate()


    def letterbox_image(self, image, size):
        iw, ih = image.size
        w, h = size
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)

        image = image.resize((nw,nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128,128,128))
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))
        return new_image


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


        if self.load_weights:
            self.yolo_model.load_weights(self.model_path, by_name=True, skip_mismatch=True)
        else:
            self.yolo_model.set_weights(self.yolo_weights)

        # self.inference_model.load_weights(self.model_path, by_name=True, skip_mismatch=True)
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
                                                          iou_threshold=self.iou)

        # logging.info('Prediction: boxes: {}, scores: {}, classes: {}'.format(np.shape(self.boxes),
        #                                                                      np.shape(self.scores),
        #                                                                      np.shape(self.classes)))


    def detect_image(self, image, image_id, *args, **kwargs):
        if self.write_down:
            self.detect_txtfile = open(os.path.join(self.pre_path, image_id + ".txt"), "w")

        start = time.time()

        # 调整图片使其符合输入要求
        new_image_size = (self.input_shape[1], self.input_shape[0])
        boxed_image = self.letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.


        # 预测结果
        # print(self.boxes, self.scores, self.classes)
        # print([image.size[1], image.size[0]])
        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        # logging.info('out boxes: {}, out_scores: {}, out_classes: {}'.format(out_boxes, out_scores, out_classes))

        # 设置字体
        font = ImageFont.truetype(font='font/simhei.ttf',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        small_pic = []
        for i, c in list(enumerate(out_classes)):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            top, left, bottom, right = box
            top = top - 5
            left = left - 5
            bottom = bottom + 5
            right = right + 5
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

            # 画框框
            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

            if self.write_down:
                self.detect_txtfile.write("%s %s %s %s %s %s\n" % (
                predicted_class, score, str(int(left)), str(int(top)), str(int(right)), str(int(bottom))))

        end = time.time()
        inference_time = np.round((end - start) * 100, 2)
        fps = int(1 / (end - start))

        # logging.info('Inference time: {} ms, FPS: {}'.format(inference_time, fps))

        if self.write_down:
            self.detect_txtfile.close()

        return image

    def close_session(self):
        self.sess.close()




class get_gt():
    def __init__(self, data_path, gt_path):
        self.data_path = data_path
        self.gt_path = gt_path
        #os.makedirs(self.gt_path, exist_ok=True)

    def __call__(self, *args, **kwargs):
        # ground_truth = []
        image_ids = open(os.path.join(self.data_path, 'ImageSets/Main/val.txt')).read().strip().split()
        for image_id in tqdm(image_ids):
            with open(os.path.join(self.gt_path, image_id + ".txt"), "w") as new_f:
                root = ET.parse(os.path.join(self.data_path, "Annotations/" + image_id + ".xml")).getroot()
                ground_truth_per_frame=[]
                for obj in root.findall('object'):
                    obj_name = obj.find('name').text
                    bndbox = obj.find('bndbox')
                    left = bndbox.find('xmin').text
                    top = bndbox.find('ymin').text
                    right = bndbox.find('xmax').text
                    bottom = bndbox.find('ymax').text
                    new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
                    # ground_truth_per_frame.append([obj_name, left, top, right, bottom])
                # print('\n')
                # print(ground_truth_per_frame)
                # ground_truth.append(ground_truth_per_frame)
        return

def read_map(path):
    fp = open(path)
    lines = fp.readlines()
    for i, line in enumerate(lines):
        if line[:3] == 'mAP':
            mAP = float(line.split()[-1].split('%')[0])
    fp.close()
    return mAP

class VOC2012mAP_Callback(tf.keras.callbacks.Callback):

    def __init__(self, data_path, command_line, visual=True, **kwargs):
        super(VOC2012mAP_Callback, self).__init__()
        self.visual = visual
        self.input_shape = (416, 416, 3)

        self.data_path = data_path
        self.annotation_path = 'Preparation/data_txt'
        self.gt_path = "input/ground-truth"
        self.pre_path = "input/detection-results"
        self.results_path = "input/"
        self.val_txt = 'face_mask_val.txt'
        self.command_line = command_line

        self.image_ids = open(os.path.join(self.data_path, 'ImageSets/Main/val.txt')).read().strip().split()
        self.excute_map_calculation()

        # Get ground truth
        get_gt(data_path=self.data_path, gt_path=self.gt_path)()

        # Prediction
        add_bias = kwargs.get('add_bias', None)
        add_bn = kwargs.get('add_bn', None)

        weight_path = kwargs.get('weight_path', None) # 'logs/yolov4/mask_recog/ep026-loss16.595-val_loss10.287.h5'
        classes_path = kwargs.get('classes_path', None) # '../Preparation/data_txt/mask_classes.txt'
        anchors_path = kwargs.get('anchors_path', None) # '../Preparation/data_txt/Mask_yolov4_anchors_416_416.txt'

        self.prediction_config = {'write_down': True,
                             'add_bn': add_bn,
                             'add_bias': add_bias,
                             'input_shape': (416, 416, 3),

                             'score': 0.5,
                             'iou': 0.3,
                             'max_boxes': 100,

                             'pre_path': self.pre_path,
                             'classes_path': classes_path,
                             'anchors_path': anchors_path,
                             'weight_path': weight_path}


        self.annotation_lines = open(os.path.join(self.annotation_path, self.val_txt)).readlines()
        self.num_val = len(self.annotation_lines)
        logger.info('Num validation: {}'.format(len(self.annotation_lines)))

        # Data processing
        # self.data_aug = Data_augmentation(input_shape=self.input_shape, visual=self.visual)

    def excute_map_calculation(self):
        if os.path.exists(self.results_path):
            shutil.rmtree(self.results_path)
            print('Cleaned the existing folder')
        os.makedirs(self.results_path)
        print('Created a new folder')
        os.makedirs(self.gt_path, exist_ok=True)
        os.makedirs(self.pre_path, exist_ok=True)


    def on_epoch_end(self, epoch, logs=None):
        pass
        #
        # yolo_weights = self.model.get_weights()
        #
        # self.get_prediction = Prediction(weights=yolo_weights, **self.prediction_config)
        #
        # for i, anno_line in enumerate(tqdm(self.annotation_lines)):
        #     line = anno_line.split()
        #     raw_image = Image.open(line[0])
        #     self.get_prediction.detect_image(image=raw_image, image_id=self.image_ids[i])
        #
        # # for yolo_weight in yolo_weights:
        # #     print(np.shape(yolo_weight), type(yolo_weight))
        # h, w, c = self.prediction_config['input_shape']
        # y_true = [Input(shape=(h // {0: 32, 1: 16, 2: 8}[l],
        #                        w // {0: 32, 1: 16, 2: 8}[l],
        #                        self.get_prediction.num_anchors // 3,
        #                        self.get_prediction.num_classes + 5)) for l in range(3)]
        #
        # # 输入为*model_body.input, *y_true
        # # 输出为model_loss
        # model_body = self.get_prediction.yolo_model
        # loss_input = [*model_body.output, *y_true]
        # #  yolo_loss(args, anchors, num_classes, ignore_thresh=.5, label_smoothing=0.1, print_loss=False):
        # model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        #                     arguments={'anchors': self.get_prediction.anchors,
        #                                'num_classes': self.get_prediction.num_classes,
        #                                'ignore_thresh': 0.5,
        #                                'label_smoothing': 0})(loss_input)
        #
        # self.model = Model([model_body.input, *y_true], model_loss)
        # self.model.set_weights(yolo_weights)
        #
        # os.system(self.command_line)
        # mAP = read_map('Callbacks/results/results.txt')
        # logs['mAP'] = mAP
        # logger.info('mAP: {} %'.format(mAP))

        # policy = mixed_precision.Policy('mixed_float16')
        # mixed_precision.set_policy(policy)
        #
        # logger.info('Compute dtype: %s' % policy.compute_dtype)
        # logger.info('Variable dtype: %s' % policy.variable_dtype)

    def on_predict_end(self, logs=None):
        yolo_weights = self.model.get_weights()

        self.get_prediction = Prediction(weights=yolo_weights, **self.prediction_config)

        for i, anno_line in enumerate(tqdm(self.annotation_lines)):
            line = anno_line.split()
            raw_image = Image.open(line[0])
            self.get_prediction.detect_image(image=raw_image, image_id=self.image_ids[i])


        os.system(self.command_line)
        mAP = read_map('Callbacks/results/results.txt')
        # logs['mAP'] = mAP
        logger.info('mAP: {} %'.format(mAP))