#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 22:10:59 2021

@author: Aaron
"""

from tensorflow import keras as K
from keras.applications import VGG16
from keras.optimizers import SGD,Adam

# Load VGG base model
vgg = VGG16(include_top = False,
            weights = 'imagenet', 
            input_tensor = None, 
            input_shape = (224,224,3), #224x224x3
            pooling = None,
            classes = 1000,
            classifier_activation="softmax") 

# Do not retrain convolutional layers
for layer in vgg.layers:
    layer.trainable = False
    
# Print model architecture
vgg.summary()
    
# Flatten layer output to add FC
x = K.layers.Flatten()(vgg.output)

