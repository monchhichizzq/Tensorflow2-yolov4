# -*- coding: utf-8 -*-
# @Time    : 2020/12/31 4:42
# @Author  : Zeqi@@
# @FileName: loss.py
# @Software: PyCharm
import logging
import math
import tensorflow as tf
from tensorflow.keras import backend as K
from Loss.ious import box_ciou
from tensorflow.keras.mixed_precision import experimental as mixed_precision

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger('Loss')
logger.setLevel(logging.DEBUG)

# ---------------------------------------------------#
#   平滑标签
# ---------------------------------------------------#
def _smooth_labels(y_true, label_smoothing):
    num_classes = tf.cast(K.shape(y_true)[-1], dtype=K.floatx())
    label_smoothing = K.constant(label_smoothing, dtype=K.floatx())
    return y_true * (1.0 - label_smoothing) + label_smoothing / num_classes

# ---------------------------------------------------#
#   将预测值的每个特征层调成真实值
# ---------------------------------------------------#
def yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):
    '''

    :param feats: tensor, yolov4 multi-scale outputs at level l
    :param anchors: tensor, anchors at level l
    :param num_classes: int, class numbers
    :param input_shape: tuple, input image shape
    :param calc_loss:
    :return:
         box_xy:
         box_wh:
         box_confidence:
         box_class_probs:
    '''

    # Number of anchors at level l, normally 3
    num_anchors = len(anchors)

    # 吧anchors转为tensor (1, 1, 1, 3, 2) (.., .., .., num_anchors, anchor_width+anchor_height)
    anchors_tensor =  tf.cast(tf.reshape(tf.constant(anchors), [1, 1, 1, num_anchors, 2]), feats.dtype)


    # 获得x，y的网格
    # feats: (batch, height, width, num_anchors*(num_classes + 5))
    grid_shape = tf.shape(feats)[1:3]  # 特征图(.., h, w, .., ..)
    grid_y = tf.tile(tf.reshape(tf.keras.backend.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
                    [1, grid_shape[1], 1, 1])
    grid_x = tf.tile(tf.reshape(tf.keras.backend.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
                     [grid_shape[0], 1, 1, 1])
    # grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
    #                 [grid_shape[0], 1, 1, 1])
    grid = tf.keras.layers.concatenate([grid_x, grid_y]) # (None, None, 1, 2)
    grid = tf.cast(grid, feats.dtype)
    # grid: (None, None, 1, 2)

    # 将 feats (batch, height, width, num_anchors*(num_classes + 5)) 变为 (batch, height, width, num_anchors， num_classes + 5)
    feats = tf.reshape(feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    # 将预测值调成真实值
    # box_xy对应框的中心点
    # box_wh对应框的宽和高
    # 对输出特征图中点(x, y)
    # 根据预测框中心点与grid中心点之间距离(sigmoid)对预测框中心点做调整
    box_xy = (tf.keras.activations.sigmoid(feats[..., :2]) + grid) / tf.cast(grid_shape[..., ::-1], feats.dtype)

    # 对预测框宽高做exp计算并且根据anchor大小缩放
    box_wh = tf.math.exp(feats[..., 2:4]) * anchors_tensor / tf.cast(input_shape[..., ::-1], feats.dtype)

    # 对输出特征图中点(x, y)中confidence取simoid，使置信度取值在0到1之间
    box_confidence = tf.keras.activations.sigmoid(feats[..., 4:5])
    # 对输出特征图中点(x, y)中类概率取simoid，使概率取值在0到1之间，一个点可以有多个物体
    box_class_probs = tf.keras.activations.sigmoid(feats[..., 5:])

    # 在计算loss的时候返回如下参数
    if calc_loss == True:
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs

def bbox_ciou(boxes1, boxes2):
    '''
    计算ciou = iou - p2/c2 - av
    :param boxes1: (8, 13, 13, 3, 4)   pred_xywh  (13, 13, 3, 4)
    :param boxes2: (8, 13, 13, 3, 4)   label_xywh (13, 13, 3, 4)
    :return:

    举例时假设pred_xywh和label_xywh的shape都是(1, 4)
    '''

    # 变成左上角坐标、右下角坐标
    boxes1_x0y0x1y1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                                 boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2_x0y0x1y1 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                                 boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)
    '''
    逐个位置比较boxes1_x0y0x1y1[..., :2]和boxes1_x0y0x1y1[..., 2:]，即逐个位置比较[x0, y0]和[x1, y1]，小的留下。
    比如留下了[x0, y0]
    这一步是为了避免一开始w h 是负数，导致x0y0成了右下角坐标，x1y1成了左上角坐标。
    '''
    boxes1_x0y0x1y1 = tf.concat([tf.minimum(boxes1_x0y0x1y1[..., :2], boxes1_x0y0x1y1[..., 2:]),
                                 tf.maximum(boxes1_x0y0x1y1[..., :2], boxes1_x0y0x1y1[..., 2:])], axis=-1)
    boxes2_x0y0x1y1 = tf.concat([tf.minimum(boxes2_x0y0x1y1[..., :2], boxes2_x0y0x1y1[..., 2:]),
                                 tf.maximum(boxes2_x0y0x1y1[..., :2], boxes2_x0y0x1y1[..., 2:])], axis=-1)

    # 两个矩形的面积
    boxes1_area = (boxes1_x0y0x1y1[..., 2] - boxes1_x0y0x1y1[..., 0]) * (
                boxes1_x0y0x1y1[..., 3] - boxes1_x0y0x1y1[..., 1])
    boxes2_area = (boxes2_x0y0x1y1[..., 2] - boxes2_x0y0x1y1[..., 0]) * (
                boxes2_x0y0x1y1[..., 3] - boxes2_x0y0x1y1[..., 1])

    # 相交矩形的左上角坐标、右下角坐标，shape 都是 (8, 13, 13, 3, 2)
    left_up = tf.maximum(boxes1_x0y0x1y1[..., :2], boxes2_x0y0x1y1[..., :2])
    right_down = tf.minimum(boxes1_x0y0x1y1[..., 2:], boxes2_x0y0x1y1[..., 2:])

    # 相交矩形的面积inter_area。iou
    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    iou = inter_area / (union_area + K.epsilon())

    # 包围矩形的左上角坐标、右下角坐标，shape 都是 (8, 13, 13, 3, 2)
    enclose_left_up = tf.minimum(boxes1_x0y0x1y1[..., :2], boxes2_x0y0x1y1[..., :2])
    enclose_right_down = tf.maximum(boxes1_x0y0x1y1[..., 2:], boxes2_x0y0x1y1[..., 2:])

    # 包围矩形的对角线的平方 C^2
    enclose_wh = enclose_right_down - enclose_left_up
    enclose_c2 = tf.math.pow(enclose_wh[..., 0], 2) + tf.math.pow(enclose_wh[..., 1], 2)

    # 两矩形中心点距离的平方 𝜌^2 (𝐴_𝑐𝑡𝑟,  𝐵_𝑐𝑡𝑟 )
    p2 = tf.math.pow(boxes1[..., 0] - boxes2[..., 0], 2) + K.pow(boxes1[..., 1] - boxes2[..., 1], 2)

    # 增加av。加上除0保护防止nan。
    atan1 = tf.atan(boxes1[..., 2] / (boxes1[..., 3] + K.epsilon()))  # w_gt/h_gt
    atan2 = tf.atan(boxes2[..., 2] / (boxes2[..., 3] + K.epsilon()))  # w/h
    v = 4.0 * tf.math.pow(atan1 - atan2, 2) / (math.pi ** 2)
    a = v / (1 - iou + v)

    ciou = iou - 1.0 * p2 / enclose_c2 - 1.0 * a * v
    return ciou

# ---------------------------------------------------#
#   用于计算每个预测框与真实框的iou
# ---------------------------------------------------#
def box_iou(b1, b2):
    '''

    :param b1: tensor, shape: (None, None, 3, 4), predict boxes
    :param b2: tensor, shape: (None, None, 3, 4), predict boxes
    :return:

    通过b1_xy 和 b1_wh计算 b1_mins_xy, b1_max_xy
    通过b2_xy 和 b2_wh计算 b2_mins_xy, b2_max_xy

    求b1_mins_xy， b2_mins_xy 中最大值 如图 b2_min
    求b1_max_xy， b2_max_xy 中最小值   如图 b1_max
    求 intersection (最大值-最小值)

           b1_min————————
                |b2_min |_____
                |   |   |    |
                |——|————|b1_max
                   |_______ |b2_max

    IOU = intersection/b1_square+b2_sqaure-intersection
    '''

    # 计算左上角的坐标和右下角的坐标
    # b1 = tf.expand_dims(b1, -2) # (None, None, 3, 1, 4)
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh / 2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    # 计算左上角和右下角的坐标
    # b2 = tf.expand_dims(b2, 0) # (1, None, None)
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh / 2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    # 计算重合面积
    intersect_mins = tf.math.maximum(b1_mins, b2_mins)
    intersect_maxes = tf.math.minimum(b1_maxes, b2_maxes)
    intersect_wh = tf.math.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)

    return iou


# ---------------------------------------------------#
#   loss值计算
# ---------------------------------------------------#
"""
大多数目标检测模型计算误差时：
直接根据预测框和真实框的中心点坐标以及宽高信息设定MSE(均方误差）损失函数或者BCE损失函数的

"""

def yolo_loss(args, anchors, num_classes, ignore_thresh=.5, label_smoothing=0.1, print_loss=False):
    policy = mixed_precision.Policy('float32')
    mixed_precision.set_policy(policy)
    logger.info('Loss Compute dtype: %s' % policy.compute_dtype)
    logger.info('Loss Variable dtype: %s' % policy.variable_dtype)

    # 一共有三层
    num_layers = len(anchors) // 3

    # 将预测结果和实际ground truth分开，args是[*model_body.output, *y_true]
    # y_true是一个列表，包含三个特征层，shape分别为(m,13,13,3,85),(m,26,26,3,85),(m,52,52,3,85)。
    # yolo_outputs是一个列表，包含三个特征层，shape分别为(m,13,13,255),(m,26,26,255),(m,52,52,255)。
    y_true = args[num_layers:]
    yolo_outputs = args[:num_layers]

    for i in range(3):
        y_true[i] = tf.cast(y_true[i], dtype=tf.float32)
        yolo_outputs[i] = tf.cast(yolo_outputs[i], dtype=tf.float32)
        logger.info('y_true: {}, type: {}'.format(y_true[i].shape, y_true[i].dtype))
        logger.info('yolo_outputs {}, type: {}'.format(yolo_outputs[i].shape, yolo_outputs[i].dtype))


    # 先验框
    # 678为142,110,  192,243,  459,401
    # 345为36,75,  76,55,  72,146
    # 012为12,16,  19,36,  40,28
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]

    # 得到input_shpae为608,608
    input_shape = tf.cast(tf.shape(yolo_outputs[0])[1:3] * 32, y_true[0].dtype)

    loss = 0

    # 取出每一张图片
    # m的值就是batch_size
    # m = K.shape(yolo_outputs[0])[0]
    # mf = K.cast(m, K.dtype(yolo_outputs[0]))
    batch_size_m = tf.shape(yolo_outputs[0])[0]
    batch_size = tf.cast(batch_size_m, yolo_outputs[0].dtype)

    # y_true是一个列表，包含三个特征层，shape分别为(m,13,13,3,85),(m,26,26,3,85),(m,52,52,3,85)。
    # yolo_outputs是一个列表，包含三个特征层，shape分别为(m,13,13,255),(m,26,26,255),(m,52,52,255)。
    for l in range(num_layers):
        # 以第一个特征层(m, 13, 13, num_anchors, (num_classes+5))为例子

        # y_true: (batch, 13, 13, num_anchors, x,y,w,h, object, class)
        # Object mask ：在该点是否存在物体
        object_mask = y_true[l][..., 4:5]

        # Truth class probabilities : 分类
        true_class_probs = y_true[l][..., 5:]

        if label_smoothing:
            true_class_probs = _smooth_labels(true_class_probs, label_smoothing)

        # 将yolo_outputs的特征层导入head进行处理
        # grid 是 特征图grid中的中心点坐标 (13, 13, 1, 2)
        # raw_pred 就是 feats
        # pred_xy 解码后的xy
        # pred_wh 解码后的wh
        grid, raw_pred, pred_xy, pred_wh = yolo_head(yolo_outputs[l],
                                                     anchors[anchor_mask[l]],
                                                     num_classes,
                                                     input_shape,
                                                     calc_loss=True)

        # 这个是解码后的预测的box的位置  (batch,13,13,3,4)
        pred_box = tf.keras.layers.concatenate([pred_xy, pred_wh])


        # 找到负样本群组，第一步是创建一个数组，[]
        ignore_mask = tf.TensorArray(y_true[0].dtype, size=1, dynamic_size=True)

        object_mask_bool = K.cast(object_mask, 'bool') # (None, 13, 13, 3, 1)


        # 对每一张图片计算ignore_mask
        def loop_body(b, ignore_mask):
            '''
            
            :param b: 
            :param ignore_mask: 
            :return: 
            '''

            # 取出第b副图内，真实存在的所有的box的参数
            # (None, None, 3, 4)
            true_box = tf.boolean_mask(y_true[l][b, ..., 0:4], object_mask_bool[b, ..., 0])

            # 计算预测结果与真实情况的iou
            # 计算的结果是每个pred_box和其它所有真实框的iou
            iou = box_iou(pred_box[b], true_box) # (None, None, 3)
            best_iou = K.max(iou, axis=-1) # (None, None)
            # ciou = bbox_ciou(pred_box[b], true_box)
            # best_iou = tf.math.reduce_max(ciou, axis=-1)

            # 如果某些预测框和真实框的重合程度小于0.5，则忽略。
            ignore_mask = ignore_mask.write(b, tf.cast(best_iou < ignore_thresh, true_box.dtype))
            return b + 1, ignore_mask

        # 遍历所有的图片
        _, ignore_mask = tf.while_loop(lambda b, *args: b < batch_size_m, loop_body, [0, ignore_mask])
        print('ignore_mask', ignore_mask)

        # 将每幅图的内容压缩，进行处理
        ignore_mask = ignore_mask.stack()
        print('ignore_mask', ignore_mask)

        ignore_mask = K.expand_dims(ignore_mask, -1)
        print('ignore_mask', ignore_mask)

        box_loss_scale = 2 - y_true[l][..., 2:3] * y_true[l][..., 3:4] # (2-wh)

######### Loss location part 1
        # Calculate ciou loss as location loss
        raw_true_box = y_true[l][..., 0:4]
        ciou = box_ciou(pred_box, raw_true_box)
        ciou_loss = object_mask * box_loss_scale * (1 - ciou)
        ciou_loss =  K.sum(ciou_loss) / batch_size
        location_loss = ciou_loss

        # 如果该位置本来有框，那么计算1与置信度的交叉熵
        # 如果该位置本来没有框，而且满足best_iou<ignore_thresh，则被认定为负样本
        # best_iou<ignore_thresh用于限制负样本数量

######### Loss confidence part 2
        # print(object_mask.shape, raw_pred[..., 4:5].shape, ignore_mask.shape)
        confidence_loss = object_mask * K.binary_crossentropy(object_mask, raw_pred[..., 4:5], from_logits=True) + \
                          (1 - object_mask) * K.binary_crossentropy(object_mask, raw_pred[..., 4:5],
                                                                    from_logits=True) * ignore_mask
######### Loss class part 3
        class_loss = object_mask * K.binary_crossentropy(true_class_probs, raw_pred[..., 5:], from_logits=True)

        confidence_loss = K.sum(confidence_loss) / batch_size
        class_loss = K.sum(class_loss) / batch_size
        loss += location_loss + confidence_loss + class_loss
        print('loss', loss.shape)
    loss = K.expand_dims(loss, axis=-1)
    print('loss', loss.shape)
    return loss

