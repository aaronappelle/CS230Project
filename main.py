#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 23:52:57 2021

@author: Aaron
"""

import numpy as np
import tensorflow as tf
from dataloader import load_task
from processdata import image_generators, split_data
from model import make_model, train_model
# import matplotlib.pyplot as plt
# from PIL import Image
from keras.preprocessing.image import array_to_img

tf.random.set_seed(42)
np.random.seed(42)

path = '/Users/Aaron/Google Drive/Documents/Stanford University/Winter 2021/CS 230/CS230 Project/Data'

task = 1

# Load Data
X_train, X_test, y_train, y_test = load_task(task,path)
imagesize = X_train[0].shape

# Plot random test image
array_to_img(X_train[np.random.randint(len(X_train))])

# Split train/val
splitsize = 0.15
shuffle = True
y_stratify = True
seed = 0

X_train, X_val, y_train, y_val = split_data(X_train, y_train, splitsize, shuffle, y_stratify, seed)

# Data augmentation
batch_size = 10
img_height = imagesize[1]
img_width = imagesize[0]
train_generator, validation_generator = image_generators(
    path, X_train, X_val, batch_size, img_height, img_width)

model = make_model()

