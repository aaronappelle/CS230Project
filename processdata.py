#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 22:00:40 2021

@author: Aaron
"""

from keras.preprocessing.image import ImageDataGenerator
# from keras.preprocessing.image import load_img, img_to_array, image_dataset_from_directory
from sklearn.model_selection import train_test_split


def split_data(X_train, y_train, splitsize, shuffle, y_stratify, seed):
    
    if y_stratify:
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=splitsize, stratify=y_train, random_state=seed)
    else:
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=splitsize, random_state=seed)

    return X_train, X_val, y_train, y_val


def image_generators(path, X_train, X_val, batch_size, image_height, image_width):

    train_datagen = ImageDataGenerator(
    featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        zca_epsilon=1e-06,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        brightness_range=None,
        shear_range=0.2,
        zoom_range=0.2,
        channel_shift_range=0.0,
        fill_mode="nearest",
        cval=0.0,
        horizontal_flip=True,
        vertical_flip=False,
        rescale=1./255,
        preprocessing_function=None,
        data_format=None,
        validation_split=0.0,
        dtype=None,
    )
     
    train_generator = train_datagen.flow_from_dataframe(
        X_train, 
        path, 
        x_col='image',
        y_col='label',
        class_mode='binary',
        target_size=(image_height, image_width),
        batch_size=batch_size
    )
    
    # For validation set, only apply scaling images
    # TODO: Or, not modify at all?
    validation_datagen = ImageDataGenerator(
        rescale=1./255
        )
    
    val_generator = validation_datagen.flow_from_dataframe(
    X_val, 
    path, 
    x_col='image',
    y_col='label',
    class_mode='binary',
    target_size=(image_height, image_width),
    batch_size=batch_size
    )
    
    return train_generator, val_generator