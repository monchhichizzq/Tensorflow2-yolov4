# -*- coding: utf-8 -*-
# @Time    : 2021/2/1 0:18
# @Author  : Zeqi@@
# @FileName: mask_data_split.py
# @Software: PyCharm

import os
import random
import numpy as np

def generate_ImageSets(dirpath):
    ids_mask = []
    ids_nomask = []
    image_ids_path = os.path.join(dirpath, 'ImageSets', 'Main')
    os.makedirs(image_ids_path, exist_ok=True)
    path = os.path.join(dirpath, 'Annotations')
    print(image_ids_path)

    for file in os.listdir(path):
        split_condition = file.split('_')[0]
        id = file.split('.')[0]

        if split_condition == 'test':
            ids_mask.append(id+ '\n')
        else:
            ids_nomask.append(id + '\n')

    num_nomask = len(ids_nomask)
    num_mask = len(ids_mask)

    train_num_nomask = int(num_nomask * 0.9)
    test_num_nomask = num_nomask - train_num_nomask
    train_num_mask = int(num_nomask * 0.9)
    test_num_mask = num_mask - train_num_mask

    print('No mask: train {}, test {}'.format(train_num_mask, test_num_mask))
    print('mask: train {}, test {}'.format(train_num_nomask, test_num_nomask))
    print('No mask samples: {}, Mask samples: {}'.format(num_nomask, num_mask))

    random.shuffle(ids_mask)
    random.shuffle(ids_nomask)

    ids_train_mask = ids_mask[:train_num_mask]
    ids_test_mask = ids_mask[train_num_mask:]
    ids_train_nomask = ids_nomask[:train_num_nomask]
    ids_test_nomask = ids_nomask[train_num_nomask:]

    ids_train_mask.extend(ids_train_nomask)
    ids_test_mask.extend(ids_test_nomask)

    with open(os.path.join(image_ids_path, 'train.txt'), 'w') as f_train:
        f_train.writelines(ids_train_mask)
        f_train.close()

    with open(os.path.join(image_ids_path, 'val.txt'), 'w') as f_val:
        f_val.writelines(ids_test_mask)
        f_val.close()

    print('Train samples: {}, Test samples: {}'.format(len(ids_train_mask), len(ids_test_mask)))


if __name__ == '__main__':
    path = 'E:/mask_detection/face_mask'
    generate_ImageSets(path)