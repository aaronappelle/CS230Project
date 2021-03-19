#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 12:15:49 2021

@author: Aaron
"""

import os
import numpy as np
import tensorflow as tf
from keras.utils import to_categorical
from dataloader import load_task
from processdata import split_data
from pseudolabeling import PseudoCallback
from model import build_model
from train import train_pseudo, train_model
from evaluate import plot_multiclass_roc, plot_performance, score_model, pred_confidence

tf.random.set_seed(42)
np.random.seed(42)

#%% Load Data

datapath = '/Users/Aaron/Google Drive/Documents/Stanford University/Winter 2021/CS 230/CS230 Project/Data'
task = 2

X_train, X_test, y_train, y_test = load_task(task,datapath)
X_train_unlabeled, X_test_unlabeled, y_test_unlabeled_1, y_test_unlabeled_2 = load_task(0,datapath)
if task == 1:
    y_test_unlabeled = y_test_unlabeled_1
elif task == 2:
    y_test_unlabeled = y_test_unlabeled_2

#%% Load Baseline Supervised Model

loadepoch = 20
# modelpath = '/Users/Aaron/Google Drive/Documents/Stanford University/Winter 2021/CS 230/CS230 Project/Results/Baseline1/Models'
modelpath = '/Users/Aaron/Google Drive/Documents/Stanford University/Winter 2021/CS 230/CS230 Project/Results/Baseline2/Models'

n_class = len(y_train[0])
basemodel = build_model(n_class)
basemodel.load_weights(modelpath + '/model_epoch' + str(loadepoch) + '.h5')

y_pred_probs = basemodel.predict(X_test_unlabeled)   # Softmax class probabilities from model
y_pred = np.argmax(y_pred_probs, axis = 1)
y_pred_oh = to_categorical(y_pred, n_class)

plot_multiclass_roc(y_pred_probs, y_pred, X_test_unlabeled, y_test_unlabeled, n_class, 'Task'+str(task)+'ROCunlabeltest.png', 
                    figsize=(9.5,5), flag=False)#, 
                    #save= 'Results/Task'+str(args.task)+'ROCunlabeltest.png')

score_model(y_test_unlabeled, y_pred_oh, y_pred_probs) #'Results/Task'+str(args.task)+'scoresunlabeltest.csv'

base_preds = y_pred

#%% Load SSL model

# loadepoch = 15
loadepoch = 5
# modelpath = '/Users/Aaron/Google Drive/Documents/Stanford University/Winter 2021/CS 230/CS230 Project/Results/SS11/Models'
modelpath = '/Users/Aaron/Google Drive/Documents/Stanford University/Winter 2021/CS 230/CS230 Project/Results/SS23/Models'

n_class = len(y_train[0])
sslmodel = build_model(n_class)
sslmodel.load_weights(modelpath + '/model_epoch' + str(loadepoch) + '.h5')

y_pred_probs = sslmodel.predict(X_test_unlabeled)   # Softmax class probabilities from model
y_pred = np.argmax(y_pred_probs, axis = 1)
y_pred_oh = to_categorical(y_pred, n_class)

plot_multiclass_roc(y_pred_probs, y_pred, X_test_unlabeled, y_test_unlabeled, n_class, 'Task'+str(task)+'ROCunlabeltest.png', 
                    figsize=(9.5,5), flag=False)#, 
                    #save= 'Results/Task'+str(args.task)+'ROCunlabeltest.png')

score_model(y_test_unlabeled, y_pred_oh, y_pred_probs) #'Results/Task'+str(args.task)+'scoresunlabeltest.csv'

ssl_preds = y_pred

#%% Compare models

y_true = np.argmax(y_test_unlabeled, axis = 1)

# Base model predicts wrongly, SSL model predicts correctly
sslbetter = np.intersect1d(np.where(base_preds != y_true), np.where(ssl_preds == y_true))

import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

for im_array in np.take(X_test_unlabeled,sslbetter,axis=0):

    imshow(im_array[:,:,::-1]) # images in bgr
    plt.show()
    plt.close()