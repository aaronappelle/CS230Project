#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 23:23:12 2021

@author: Aaron
"""

import numpy as np
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
        # labeled_unlabeledの作成
        # (X_train, y_train), (self.X_test, self.y_test) = cifar10.load_data()
        # indices = np.arange(X_train.shape[0])
        # np.random.shuffle(indices)
        # self.X_train_labeled = X_train[indices[:n_labeled_sample]]
        # self.y_train_labeled = y_train[indices[:n_labeled_sample]]
        # self.X_train_unlabeled = X_train[indices[n_labeled_sample:]]
        # self.y_train_unlabeled_groundtruth = y_train[indices[n_labeled_sample:]]
        
        self.X_train_labeled = X_train
        self.y_train_labeled = y_train
        self.X_train_unlabeled = X_train_unlabeled
        self.X_test = X_test
        self.y_test = y_test
        
        # unlabeled prediction
        # self.y_train_unlabeled_prediction = np.random.randint(
        #     10, size=(self.y_train_unlabeled_groundtruth.shape[0], 1))
        self.y_train_unlabeled_prediction = np.random.randint(
            self.n_classes, size=(self.X_train_unlabeled.shape[0], 1))
        # steps_per_epoch
        self.train_steps_per_epoch = X_train.shape[0] // batch_size
        self.test_stepes_per_epoch = self.X_test.shape[0] // batch_size
        # unlabeled
        self.alpha_t = 0.0
        # labeled/unlabeled accuracy
        # self.unlabeled_accuracy = [] #can't record this bc no labels...
        self.labeled_accuracy = []

    def train_mixture(self):
        # Combine all examples and flag whether it is labeled or unlabeled
        # X_train_join = np.r_[self.X_train_labeled, self.X_train_unlabeled]
        X_train_join = np.vstack((self.X_train_labeled,self.X_train_unlabeled))
        # y_train_join = np.r_[self.y_train_labeled, self.y_train_unlabeled_prediction]
        y_train_join = np.vstack((self.y_train_labeled, to_categorical(self.y_train_unlabeled_prediction)))
        flag_join = np.r_[np.repeat(0.0, self.X_train_labeled.shape[0]),
                         np.repeat(1.0, self.X_train_unlabeled.shape[0])].reshape(-1,1)
        indices = np.arange(flag_join.shape[0])
        np.random.shuffle(indices)
        return X_train_join[indices], y_train_join[indices], flag_join[indices]

    def train_generator(self):
        # Generate batches of training data (mixed labeled/unlabeled)
        while True:
            X, y, flag = self.train_mixture()
            n_batch = X.shape[0] // self.batch_size
            for i in range(n_batch):
                # normalize images values
                X_batch = (X[i*self.batch_size:(i+1)*self.batch_size]/255.0).astype(np.float32)
                # y_batch = to_categorical(y[i*self.batch_size:(i+1)*self.batch_size], self.n_classes)
                y_batch = y[i*self.batch_size:(i+1)*self.batch_size] # y is already categorical...?
                y_batch = np.c_[y_batch, flag[i*self.batch_size:(i+1)*self.batch_size]]
                yield X_batch, y_batch

    def test_generator(self):
        while True:
            indices = np.arange(self.y_test.shape[0])
            np.random.shuffle(indices)
            for i in range(len(indices)//self.batch_size):
                current_indices = indices[i*self.batch_size:(i+1)*self.batch_size]
                # normalize images values
                X_batch = (self.X_test[current_indices] / 255.0).astype(np.float32)
                # y_batch = to_categorical(self.y_test[current_indices], self.n_classes)
                y_batch = self.y_test[current_indices]
                y_batch = np.c_[y_batch, np.repeat(0.0, y_batch.shape[0])] # flagは0とする
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
        # alpha(t)の更新
        # if epoch < 10:
        if epoch < self.a_limits[0]:    
            self.alpha_t = 0.0
        # elif epoch >= 70:
        elif epoch >= self.a_limits[-1]:    
            self.alpha_t = 3.0
        else:
            # self.alpha_t = (epoch - 10.0) / (70.0-10.0) * 3.0
            self.alpha_t = (epoch - self.a_limits[0]) / (self.a_limits[-1]-self.a_limits[0]) * 3.0
        # unlabeled predictions from most probable class
        self.y_train_unlabeled_prediction = np.argmax(
            self.model.predict(self.X_train_unlabeled), axis=-1,).reshape(-1, 1)
        y_train_labeled_prediction = np.argmax(
            self.model.predict(self.X_train_labeled), axis=-1).reshape(-1, 1)
        # ground-truth
        # self.unlabeled_accuracy.append(np.mean(
        #     self.y_train_unlabeled_groundtruth == self.y_train_unlabeled_prediction))
        self.labeled_accuracy.append(np.mean(
            self.y_train_labeled == y_train_labeled_prediction))
        # print("labeled / unlabeled accuracy : ", self.labeled_accuracy[-1],
        #     "/", self.unlabeled_accuracy[-1])
        print("Accuracy : ", self.labeled_accuracy[-1])

    def on_train_end(self, logs):
        y_true = np.ravel(np.argmax(self.y_test,axis=-1))
        emb_model = Model(self.model.input, self.model.layers[-2].output)
        embedding = emb_model.predict(self.X_test / 255.0)
        proj = TSNE(n_components=2).fit_transform(embedding)
        cmp = plt.get_cmap("tab10")
        plt.figure()
        # for i in range(10):
        for i in range(self.n_classes):    
            select_flag = y_true == i
            plt_latent = proj[select_flag, :]
            plt.scatter(plt_latent[:,0], plt_latent[:,1], color=cmp(i), marker=".")
        plt.savefig(f"result_pseudo/embedding_{self.n_labeled_sample:05}.png")


# def train(n_labeled_data):
#     model = create_cnn()
    
#     pseudo = PseudoCallback(model, n_labeled_data, min(512, n_labeled_data))
#     model.compile("adam", loss=pseudo.loss_function, metrics=[pseudo.accuracy])

#     if not os.path.exists("result_pseudo"):
#         os.mkdir("result_pseudo")

#     hist = model.fit_generator(pseudo.train_generator(), steps_per_epoch=pseudo.train_steps_per_epoch,
#                                validation_data=pseudo.test_generator(), callbacks=[pseudo],
#                                validation_steps=pseudo.test_stepes_per_epoch, epochs=1).history
#     hist["labeled_accuracy"] = pseudo.labeled_accuracy
#     hist["unlabeled_accuracy"] = pseudo.unlabeled_accuracy

#     with open(f"result_pseudo/history_{n_labeled_data:05}.dat", "wb") as fp:
#         pickle.dump(hist, fp)