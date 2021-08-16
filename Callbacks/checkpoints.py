# -*- coding: utf-8 -*-
# @Time    : 2020/12/31 4:44
# @Author  : Zeqi@@
# @FileName: checkpoints.py
# @Software: PyCharm

import warnings
from tensorflow.keras.callbacks import Callback
import numpy as np

class ModelCheckpoint(Callback):
    def __init__(self, filepath, monitor='val_loss', verbose=1,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super(ModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0
        self.history = []
        self.history_save_file = 'train_history_sgd.npy'

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    # def on_train_batch_end(self, batch, logs=None):
    #     current_lr = self.model.optimizer.lr.numpy()
    #     print('current lr: ', current_lr, batch)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current_lr = self.model.optimizer.lr.numpy()
        print('current lr: ', current_lr)
        train_loss, val_loss = logs['loss'], logs['val_loss']
        self.history.append([epoch, train_loss, val_loss, current_lr])
        np.save(self.history_save_file, self.history)

        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            self.model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve' %
                                  (epoch + 1, self.monitor))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    # self.model.summary()
                    # for layer in self.model.layers:
                    #     print(layer.name, np.shape(layer.get_weights()))
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True)

