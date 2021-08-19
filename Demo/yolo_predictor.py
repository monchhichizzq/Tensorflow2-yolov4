# -*- coding: utf-8 -*-
# @Time    : 2021/8/12 3:37
# @Author  : Zeqi@@
# @FileName: yolo_predictor.py
# @Software: PyCharm

import os
import colorsys
import numpy as np
from timeit import default_timer as timer

from PIL import Image
from Models.yolov4 import yolov4
from Models.yolo_val import yolo_eval
from Callbacks.plot_utils import plot_one_box

import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.compat.v1.keras import backend as K

def letterbox_image(image, size):
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image

class YOLO(object):
    _defaults = {
        "model_path"        : '../Trainer/logs/yolov4_voc_weights_416_fp16/ep190-loss25.458-val_loss13.749.h5',
        "classes_path"       : '../Preparation/data_txt/voc_obj/voc_names.txt',
        "anchors_path"      : '../Preparation/data_txt/voc_obj/yolov4_anchors_416_416.txt',
        "score"             : 0.2,
        "iou"               : 0.3,
        "eager"             : False,
        "max_boxes"         : 100,
        "model_image_size"  : (416, 416)
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        if not self.eager:
            tf.compat.v1.disable_eager_execution()
            self.sess = K.get_session()
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'
        
        # 计算anchor数量
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)

        # 载入模型，如果原来的模型里已经包括了模型结构则直接载入。
        # 否则先构建模型再载入

        inputs = Input(shape=(416, 416, 3))
        self.yolo_model = yolov4(num_anchors=num_anchors // 3, num_classes=num_classes, add_bias=False, add_bn=True)(inputs)
        self.yolo_model.summary()
        self.yolo_model.load_weights(self.model_path, by_name=True, skip_mismatch=True)
        # self.yolo_model.load_weights(self.model_path)
        print('{} model, anchors, and classes loaded.'.format(model_path))

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

        if self.eager:
            self.input_image_shape = Input([2,],batch_size=1)
            inputs = [*self.yolo_model.output, self.input_image_shape]
            outputs = Lambda(yolo_eval, output_shape=(1,), name='yolo_eval',
                arguments={'anchors': self.anchors, 'num_classes': len(self.class_names), 'image_shape': self.model_image_size,
                'score_threshold': self.score, 'eager': True, 'max_boxes': self.max_boxes})(inputs)
            self.yolo_model = Model([self.yolo_model.input, self.input_image_shape], outputs)
        else:
            self.input_image_shape = K.placeholder(shape=(2, ))

            self.boxes, self.scores, self.classes = yolo_eval(self.yolo_model.output, self.anchors,
                    num_classes, self.input_image_shape, max_boxes=self.max_boxes,
                    score_threshold=self.score, iou_threshold=self.iou, letterbox_image=True)

        # print('Prediction: boxes: {}, scores: {}, classes: {}'.format(np.shape(self.boxes), np.shape(self.scores), np.shape(self.classes)))

    def detect_image(self, image):
        # start = timer()
        raw_image = image.copy()
        raw_image = np.array(raw_image, dtype=np.uint8)

        # 调整图片使其符合输入要求
        new_image_size = (self.model_image_size[1],self.model_image_size[0])
        boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        start = timer()
        if self.eager:
            # 预测结果
            input_image_shape = np.expand_dims(np.array([image.size[1], image.size[0]], dtype='float32'), 0)
            out_boxes, out_scores, out_classes = self.yolo_model.predict([image_data, input_image_shape]) 
        else:
            # 预测结果

            out_boxes, out_scores, out_classes = self.sess.run(
                [self.boxes, self.scores, self.classes],
                feed_dict={
                    self.yolo_model.input: image_data,
                    self.input_image_shape: [image.size[1], image.size[0]],
                    K.learning_phase(): 0
                })

            # print('out boxes: {}, out_scores: {}, out_classes: {}'.format(out_boxes, out_scores, out_classes))
        end = timer()

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

            plot_one_box(raw_image, [int(left), int(top), int(right), int(bottom)],
                        class_names=self.class_names,
                        label=predicted_class + ', {:.2f}%'.format(np.round(float(score) * 100,2)),
                        color=self.colors[int(c)-1])

        inference_time = np.round((end-start) * 1000, 2)
        fps = int(1/(end-start))
        print('Inference time: {} ms, FPS: {}'.format(inference_time, fps))
        return raw_image

    def close_session(self):
        self.sess.close()
