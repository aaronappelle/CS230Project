#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 22:10:59 2021

@author: Aaron
"""

from tensorflow import keras
from keras.applications import vgg16
from keras.optimizers import SGD,Adam

vgg = vgg16(include_top = True,
            weights = 'imagenet', 
            input_tensor = None, 
            input_shape = None, #224x224x3
            pooling = None,
            classes = 1000,
            classifier_activation="softmax") 