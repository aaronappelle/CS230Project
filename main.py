#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 13:04:22 2021

@author: Aaron
"""

import argparse
import numpy as np
import tensorflow as tf
from keras.utils import to_categorical
from dataloader import load_task
from processdata import split_data
from pseudolabeling import PseudoCallback
from model import build_model
from train import train_pseudo, train_model
from evaluate import plot_multiclass_roc, plot_performance, score_model, pred_confidence
# import matplotlib.pyplot as plt

tf.random.set_seed(42)
np.random.seed(42)

#%%
def main():
    
    #%% Command line arguments parser
    parser = argparse.ArgumentParser(description = 'Keras Pseudo-Label Training')
    parser.add_argument('--task', 
                        default = 1,
                        type = int,
                        choices = [1, 2], 
                        help = 'Task 1: Scene Level (Pixel, Object, Structure), Task 2: Damage State (Damaged, Undamaged)')
    parser.add_argument('--semisupervised',
                        default = False,
                        action='store_true',
                        help = 'True/False use unlabeled images for pseudo-label training')
    parser.add_argument('--path', 
                        default = '/home/ubuntu', 
                        type = str,
                        help = 'Location of folders containing datasets for each task')
    parser.add_argument('--val_split',
                        default = 0.1,
                        type = float,
                        help = 'Proportion of (labeled) training set to use as validation')
    parser.add_argument('--batch_size',
                        default = 32,
                        type = int,
                        help = 'Number of examples per training batch')
    parser.add_argument('--epochs',
                        default = 1,
                        type = int,
                        help = 'Number of epochs to train from data')
    parser.add_argument('--lr',
                        default = 1e-4,
                        type = float,
                        help = 'Adam optimizer learning rate')
    parser.add_argument('--alpha_range',
                        nargs = 2,
                        type = int,
                        default = [2,5],
                        help = 'List of length two defining the epoch to start\
                                including pseudo-labels in loss function, and\
                                the epoch to end the ramp-up of loss weighting')
    args = parser.parse_args()    
    
    #%% Load Data
    X_train, X_test, y_train, y_test = load_task(args.task,args.path)
    
    if args.semisupervised:
        X_train_unlabeled, X_test_unlabeled, y_test_unlabeled_1, y_test_unlabeled_2 = load_task(0,args.path)
        if args.task == 1:
            y_test_unlabeled = y_test_unlabeled_1
        elif args.task == 2:
            y_test_unlabeled = y_test_unlabeled_2
    #%% Split Data
    
    # Shuffle and shorten (for speed, for now)
    indices = np.random.randint(0,20000,(64,))
    X_train = X_train.take(indices,axis=0)
    y_train = y_train.take(indices,axis=0)
    indices = np.random.randint(0,2500,(64,))
    X_val = X_test.take(indices,axis=0)
    y_val = y_test.take(indices,axis=0)
    indices = np.random.randint(0,4000,(64,))
    X_train_unlabeled = X_train_unlabeled.take(indices,axis=0)
    
    # # Plot random test image
    # array_to_img(X_train[np.random.randint(len(X_train))][:,:,::-1]) # images in BGR!
    # array_to_img(X_train_unlabeled[np.random.randint(len(X_train_unlabeled))][:,:,::-1]) # images in BGR!
    
    
    #%% Train Model
    
    n_class = len(y_train[0])
    
    model = build_model(n_class)
    
    if args.semisupervised:
        
        shuffle = True
        y_stratify = True
        seed = 0
    
        X_train, X_val, y_train, y_val = split_data(X_train, y_train, args.val_split, shuffle, y_stratify, seed)
        
        pseudo = PseudoCallback(model, X_train, y_train, X_train_unlabeled,
                         X_val, y_val, args.batch_size, args.alpha_range)
        
        hist = train_pseudo(model, pseudo, args.epochs, args.lr)

        # return model, hist, pseudo
    
    else:
        
        hist = train_model(model, X_train, y_train, args.val_split, args.batch_size, args.epochs)
        # return model, hist
        
    #%% Evaluate Model 
    
    plot_performance(hist, save = 'Results/Task'+str(args.task)+'history.png')
    
    # Test set predictions (labeled)
    y_pred_probs = model.predict(X_test)   # Softmax class probabilities from model
    y_pred = np.argmax(y_pred_probs, axis = 1)
    y_pred_oh = to_categorical(y_pred)
    
    if args.task == 1:
        title = 'Task 1: Scene Level'    
    elif args.task == 2:
        title = 'Task 2: Damage State'
    
    n_class = len(y_test[0])
    
    plot_multiclass_roc(y_pred_oh, y_pred, X_test, y_test, n_class, title, 
                        figsize=(9.5,5), flag=False, 
                        save='Results/Task'+str(args.task)+'ROClabeltest.png')
    
    score_model(y_test, y_pred_oh, save = 'Results/Task'+str(args.task)+'scoreslabeltest.csv')
    
    if args.semisupervised:
        
        # Test set predictions (labeled)
        y_pred_probs = model.predict(X_test_unlabeled)   # Softmax class probabilities from model
        y_pred = np.argmax(y_pred_probs, axis = 1)
        y_pred_oh = to_categorical(y_pred)
        
        plot_multiclass_roc(y_pred_oh, y_pred, X_test_unlabeled, y_test_unlabeled, n_class, title, 
                            figsize=(9.5,5), flag=False, 
                            save='Results/Task'+str(args.task)+'ROCunlabeltest.png')
        
        score_model(y_test, y_pred_oh, save = 'Results/Task'+str(args.task)+'scoresunlabeltest.csv')


#%%
    
if __name__ == '__main__':
    
    main()