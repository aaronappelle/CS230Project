#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 13:04:22 2021

@author: Aaron
"""

import argparse
import numpy as np
import tensorflow as tf
from dataloader import load_task
from processdata import split_data
from pseudolabeling import PseudoCallback
from model import build_model
from train import train_pseudo
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
                        default = '/Users/Aaron/Google Drive/Documents/Stanford University/Winter 2021/CS 230/CS230 Project/Data', 
                        type = str,
                        help = 'Location of folders containing datasets for each task')
    parser.add_argument('--val_split',
                        default = 0.15,
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
        X_train_unlabeled, _, _, _ = load_task(0,args.path)
    
    #%% Split Data

    shuffle = True
    y_stratify = True
    seed = 0
    
    X_train, X_val, y_train, y_val = split_data(X_train, y_train, args.val_split, shuffle, y_stratify, seed)
    
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
    
    pseudo = PseudoCallback(model, X_train, y_train, X_train_unlabeled,
                     X_val, y_val, args.batch_size, args.alpha_range)
    
    hist = train_pseudo(model, pseudo, args.epochs, args.lr)

    return model, pseudo, hist
    
    
if __name__ == '__main__':
    
    main()