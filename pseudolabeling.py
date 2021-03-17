#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 23:23:12 2021

@author: Aaron
"""

import numpy as np
import os
from keras.models import Model
from keras.callbacks import Callback
from keras.utils import to_categorical
from keras.objectives import categorical_crossentropy
from keras.metrics import categorical_accuracy
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

class PseudoCallback(Callback):
    def __init__(self, model, X_train, y_train, X_train_unlabeled,
                 X_test, y_test, batch_size, alpha_limits):
        self.n_labeled_sample = X_train.shape[0]
        self.batch_size = batch_size
        self.model = model
        self.n_classes = y_train[0].shape[0]
        self.a_limits = alpha_limits # start and stop of alpha coef for unlabeled images
        
        self.X_train_labeled = X_train
        self.y_train_labeled = y_train
        self.X_train_unlabeled = X_train_unlabeled
        self.X_test = X_test
        self.y_test = y_test

        # unlabeled
        self.y_train_unlabeled_prediction = to_categorical(np.random.randint(
            self.n_classes, size=(self.X_train_unlabeled.shape[0], 1)), self.n_classes)
        
        # steps_per_epoch
        self.train_steps_per_epoch = X_train.shape[0] // batch_size
        self.test_steps_per_epoch = self.X_test.shape[0] // batch_size
        # unlabeled
        self.alpha_t = 0.0

    def train_mixture(self):

        X_train_join = np.r_[self.X_train_labeled, self.X_train_unlabeled]
        y_train_join = np.r_[self.y_train_labeled, self.y_train_unlabeled_prediction]
        flag_join = np.r_[np.repeat(0.0, self.X_train_labeled.shape[0]),
                         np.repeat(1.0, self.X_train_unlabeled.shape[0])].reshape(-1,1)
        indices = np.arange(flag_join.shape[0])
        # np.random.shuffle(indices)
        # return X_train_join[indices], y_train_join[indices], flag_join[indices]
        np.save('Xtrainjoin.npy',X_train_join[indices])
        np.save('ytrainjoin.npy',y_train_join[indices])
        np.save('flagjoin.npy',flag_join[indices])

    def train_generator(self):
        while True:
            # X, y, flag = self.train_mixture()
            
            self.train_mixture()
            
            X = np.load('Xtrainjoin.npy', mmap_mode = 'r')
            y = np.load('ytrainjoin.npy', mmap_mode = 'r')
            flag = np.load('flagjoin.npy', mmap_mode = 'r')
            
            n_batch = X.shape[0] // self.batch_size
            for i in range(n_batch):
                X_batch = X[i*self.batch_size:(i+1)*self.batch_size]
                y_batch = y[i*self.batch_size:(i+1)*self.batch_size]
                y_batch = np.c_[y_batch, flag[i*self.batch_size:(i+1)*self.batch_size]]
                yield X_batch, y_batch

    def test_generator(self):
        while True:
            indices = np.arange(self.y_test.shape[0])
            # np.random.shuffle(indices)
            for i in range(len(indices)//self.batch_size):
                current_indices = indices[i*self.batch_size:(i+1)*self.batch_size]
                X_batch = self.X_test[current_indices]
                y_batch = self.y_test[current_indices]
                y_batch = np.c_[y_batch, np.repeat(0.0, y_batch.shape[0])]
                yield X_batch, y_batch

    def loss_function(self, y_true, y_pred):
        y_true_item = y_true[:, :self.n_classes]
        unlabeled_flag = y_true[:, self.n_classes]
        entropies = categorical_crossentropy(y_true_item, y_pred)
        coefs = 1.0-unlabeled_flag + self.alpha_t * unlabeled_flag # 1 if labeled, else alpha_t
        return coefs * entropies

    def accuracy(self, y_true, y_pred):
        y_true_item = y_true[:, :self.n_classes]
        return categorical_accuracy(y_true_item, y_pred)

    def on_epoch_end(self, epoch, logs):
        if epoch < self.a_limits[0]:    
            self.alpha_t = 0.0
        elif epoch >= self.a_limits[-1]:    
            self.alpha_t = 3.0
        else:
            self.alpha_t = (epoch - self.a_limits[0]) / (self.a_limits[-1]-self.a_limits[0]) * 3.0

        if not os.path.exists("Models"):
            os.mkdir("Models")
        
        self.model.save(f'Models/model_epoch{epoch}.h5')

    def on_train_end(self, logs):
        y_true = np.ravel(np.argmax(self.y_test, axis = 1))
        emb_model = Model(self.model.input, self.model.layers[-2].output)
        embedding = emb_model.predict(self.X_test)
        proj = TSNE(n_components=2).fit_transform(embedding)
        cmp = plt.get_cmap("tab10")
        plt.figure()
        for i in range(10):
            select_flag = y_true == i
            plt_latent = proj[select_flag, :]
            plt.scatter(plt_latent[:,0], plt_latent[:,1], color=cmp(i), marker=".")
            
        if not os.path.exists("Results"):
            os.mkdir("Results")
            
        plt.savefig(f"Results/embedding_{self.n_labeled_sample:05}.png")