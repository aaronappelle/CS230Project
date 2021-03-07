#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 13:01:59 2021

@author: Aaron
"""

import pickle, os, zipfile, glob

# Pseud-labels
def train_pseudo(model, pseudo, epochs):
    
    model.compile("adam", loss=pseudo.loss_function, metrics=[pseudo.accuracy])

    if not os.path.exists("result_pseudo"):
        os.mkdir("result_pseudo")

    # hist = model.fit_generator(pseudo.train_generator(), steps_per_epoch=pseudo.train_steps_per_epoch,
    #                            validation_data=pseudo.test_generator(), callbacks=[pseudo],
    #                            validation_steps=pseudo.test_stepes_per_epoch, epochs=1).history
    
    hist = model.fit(pseudo.train_generator(), steps_per_epoch=pseudo.train_steps_per_epoch,
                               validation_data=pseudo.test_generator(), callbacks=[pseudo],
                               validation_steps=pseudo.test_stepes_per_epoch, epochs=epochs).history
    
    hist["labeled_accuracy"] = pseudo.labeled_accuracy
    # hist["unlabeled_accuracy"] = pseudo.unlabeled_accuracy

    with open("result_pseudo/history.dat", "wb") as fp:
        pickle.dump(hist, fp)

    return hist

# Train model without data augmentation, split val set w/ Keras model.fit()
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

# Train model using data augmentation
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

# Train model without data augmentation but using manually partitioned validation set
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