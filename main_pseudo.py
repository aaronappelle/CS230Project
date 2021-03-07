#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 13:04:22 2021

@author: Aaron
"""

import numpy as np
import tensorflow as tf
from dataloader import load_task
from processdata import image_generators, split_data
from pseudolabeling import PseudoCallback
from model import build_model
from train import train_pseudo
# import matplotlib.pyplot as plt
from PIL import Image
import PIL.ImageOps
from keras.preprocessing.image import array_to_img

tf.random.set_seed(42)
np.random.seed(42)

path = '/Users/Aaron/Google Drive/Documents/Stanford University/Winter 2021/CS 230/CS230 Project/Data'

#%% Load Data

task = 1
X_train, X_test, y_train, y_test = load_task(task,path)
imagesize = X_train[0].shape

task = 0
X_train_unlabeled, _, _, _ = load_task(task,path)

# Shuffle and shorten (for speed, for now)
indices = np.random.randint(0,20000,(64,))
X_train = X_train.take(indices,axis=0)
y_train = y_train.take(indices,axis=0)
indices = np.random.randint(0,2500,(64,))
X_test = X_test.take(indices,axis=0)
y_test = y_test.take(indices,axis=0)
indices = np.random.randint(0,4000,(64,))
X_train_unlabeled = X_train_unlabeled.take(indices,axis=0)

#%% Plot random test image
array_to_img(X_train[np.random.randint(len(X_train))][:,:,::-1]) # images in BGR!
array_to_img(X_train_unlabeled[np.random.randint(len(X_train_unlabeled))])


#%% Train Model

n_class = len(y_train[0])
model = build_model(n_class)

val_split = 0.15
batch_size = 32
epochs = 10
alpha_limits = [3,10]

pseudo = PseudoCallback(model, X_train, y_train, X_train_unlabeled,
                 X_test, y_test, batch_size, alpha_limits)


hist = train_pseudo(model, pseudo, epochs)