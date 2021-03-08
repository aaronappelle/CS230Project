#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 22:10:59 2021

@author: Aaron

WORKS REFERENCED:
    Akhil Jhanwar: https://medium.com/analytics-vidhya/cnn-transfer-learning-with-vgg16-using-keras-b0226c0805bd
    TheBinaryNotes: https://thebinarynotes.com/transfer-learning-keras-vgg16/
    
"""

from keras.applications import VGG16
from keras import layers
from keras.models import Model
from keras.layers import Dropout, Flatten, Dense, GlobalMaxPooling2D

def build_model(n_class):
    
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
    # vgg.summary()
        
    # Get last layer
    last_layer = vgg.get_layer('block5_pool')
    
    # Add new FC layers
    x = GlobalMaxPooling2D()(last_layer.output)
    # x = Flatten()(vgg.output)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = layers.Dense(n_class, activation='softmax')(x)
    
    model = Model(inputs = vgg.input, outputs = x)
    
    # model.compile(loss = 'categorical_crossentropy',
    #               optimizer = Adam(learning_rate = lr),
    #               # optimizer = SGD(lr = 1e-4, momentum = 0.9),
    #               metrics = ['accuracy'])
    
    # model.summary()
    
    return model