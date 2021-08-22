#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@auther    : zzhu
@time      : 21/7/21 7:14 PM
@fileName  : plot_history.py
'''

import os
import numpy as np
import matplotlib.pyplot as plt

class Train_Process_Visualizer():
    def __init__(self, **kwargs):
        # self.model_folder = kwargs.get('model_folder', None)
        save = kwargs.get('save_npy', 'train_history.npy')
        self.save_folder = kwargs.get('save_folder', 'save_folder')
        os.makedirs(self.save_folder, exist_ok=True)
        self.history_path = os.path.join(self.save_folder, save)

        # self.epochs = []
        # self.lrs = []
        # self.losses = []
        # self.iou_losses = []
        # self.conf_losses = []
        # self.obj_losses = []

    def read_data(self):
        history_data = np.load(self.history_path, allow_pickle=True)
        self.epochs = history_data[..., 0]
        self.lrs = history_data[..., 3]
        self.losses = history_data[..., 1]
        self.val_losses = history_data[..., 2]
   
    def sort_data(self, data_list):
        data_list = sorted(data_list, key=lambda x: x[0])
        sort_data_list = [m[1] for m in data_list]
        epochs_list = [m[0] for m in data_list]
        return epochs_list, sort_data_list

    def plot_lr(self):
        save_fig = os.path.join(self.save_folder, 'train_lrs.png')
        plt.plot(self.epochs, self.lrs)
        plt.xlabel('Epoch')
        plt.ylabel('lr')
        plt.grid()
        plt.savefig(save_fig, dpi=600)
        plt.show()

    def plot_losses(self):
        save_fig = os.path.join(self.save_folder, 'train_losses.png')
        plt.plot(self.epochs, self.losses, label='train_loss')
        plt.plot(self.epochs, self.val_losses, label='val_loss')
        plt.legend()
        plt.grid()
        plt.ylim(0, 50)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(save_fig, dpi=600)
        plt.show()

    def __call__(self, target_metric):
        self.target_metric = target_metric
        self.read_data()
        if self.target_metric == 'lr':
            self.plot_lr()
        if self.target_metric == 'losses':
            self.plot_losses()


if __name__ == '__main__':
    
    v = Train_Process_Visualizer(save_folder = '../Trainer',
                                save_npy = 'train_history_sgd.npy',)
    v(target_metric = 'losses')
    v(target_metric = 'lr')
