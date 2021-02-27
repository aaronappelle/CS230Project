#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 22:10:59 2021

@author: Aaron
"""

from tensorflow import keras as K
from keras.applications import VGG16
from keras.optimizers import SGD,Adam
from keras import layers
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, GlobalMaxPooling2D

def make_model():
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
        
    # nPre = 15
    # for layer in vgg.layers[:nPre]:
    #     layer.trainable = False
    # for layer in vgg.layers[nPre:]
    
    
    # Print model architecture
    vgg.summary()
        
    # Get last layer
    last_layer = vgg.get_layer('block5_pool')
    
    # Add new FC layers
    x = GlobalMaxPooling2D()(last_layer.output)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = layers.Dense(1, activation='sigmoid')(x)
    
    model = Model(vgg.input, x)
    
    model.compile(loss = 'binary_crossentroypy',
                  optimizer = Adam(learning_rate = 1e-4),
                  # optimizer = SGD(lr = 1e-4, momentum = 0.9),
                  metrics = ['accuracy'])
    
    model.summary()
    
    return model

def train_model(model, Xtrain, Xvalid, ytrain, yvalid):
    n_train = len()