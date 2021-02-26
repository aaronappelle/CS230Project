#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 14:31:38 2021

@author: Aaron
"""

import numpy as np

def load_task(task,path):
    if task == 1:
        Xtrain = np.load(path + '/task1/task1_X_train.npy', mmap_mode = 'r')
        ytrain = np.load(path + '/task1/task1_y_train.npy', mmap_mode = 'r')
        Xtest = np.load(path + '/task1/task1_X_test.npy', mmap_mode = 'r')
        ytest = np.load(path + '/task1/task1_y_test.npy', mmap_mode = 'r')
    elif task == 2:
        Xtrain = np.load(path + '/task2/task2_X_train.npy', mmap_mode = 'r')
        ytrain = np.load(path + '/task2/task2_y_train.npy', mmap_mode = 'r')
        Xtest = np.load(path + '/task2/task2_X_test.npy', mmap_mode = 'r')
        ytest = np.load(path + '/task2/task2_y_test.npy', mmap_mode = 'r')
    return Xtrain, Xtest, ytrain, ytest