#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 22:10:59 2021

@author: Aaron

WORKS REFERENCED:
    Akhil Jhanwar: https://medium.com/analytics-vidhya/cnn-transfer-learning-with-vgg16-using-keras-b0226c0805bd
    TheBinaryNotes: https://thebinarynotes.com/transfer-learning-keras-vgg16/
    
"""

from tensorflow import keras as K
from keras.applications import VGG16
from keras.optimizers import SGD,Adam
from keras import layers
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, GlobalMaxPooling2D

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
    vgg.summary()
        
    # Get last layer
    last_layer = vgg.get_layer('block5_pool')
    
    # Add new FC layers
    x = GlobalMaxPooling2D()(last_layer.output)
    # x = Flatten()(vgg.output)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = layers.Dense(n_class, activation='softmax')(x)
    
    model = Model(inputs = vgg.input, outputs = x)
    
    model.compile(loss = 'categorical_crossentropy',
                  optimizer = Adam(learning_rate = 1e-4),
                  # optimizer = SGD(lr = 1e-4, momentum = 0.9),
                  metrics = ['accuracy'])
    
    model.summary()
    
    return model

def train_gen_model(model, X_train, X_val, y_train, y_val, train_generator, val_generator, batch_size, epochs):
    n_train = len(X_train)
    n_val = len(X_val)
    
    hist = model.fit_generator(
        train_generator,
        epochs = epochs,
        validation_data = val_generator,
        validation_steps = n_val//batch_size,
        steps_per_epoch = n_train//batch_size
        )
    
    return hist

def train_model_val(model, X_train, X_val, y_train, y_val, batch_size, epochs):
    n_train = len(X_train)
    n_val = len(X_val)
    
    hist = model.fit(
        x = X_train,
        y = y_train,
        batch_size = batch_size,
        epochs = epochs,
        validation_data = X_val,
        validation_steps = n_val//batch_size,
        steps_per_epoch = n_train//batch_size
        )
    
    return hist

def train_model(model, X_train, y_train, val_split, batch_size, epochs):
    n_train = len(X_train)
    
    # TODO: try class_weight
    
    hist = model.fit(
        x = X_train,
        y = y_train,
        validation_split= val_split,
        batch_size = batch_size,
        epochs = epochs,
        steps_per_epoch = n_train//batch_size
        )
    
    return hist