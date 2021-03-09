#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 14:31:38 2021

@author: Aaron
"""

import numpy as np

def load_task(task,path):
    if task == 1:
        # Scene Level
        X_train = np.load(path + '/task1/task1_X_train.npy', mmap_mode = 'r')
        y_train = np.load(path + '/task1/task1_y_train.npy', mmap_mode = 'r')
        X_test = np.load(path + '/task1/task1_X_test.npy', mmap_mode = 'r')
        y_test = np.load(path + '/task1/task1_y_test.npy', mmap_mode = 'r')
    elif task == 2:
        # Damage State
        X_train = np.load(path + '/task2/task2_X_train.npy', mmap_mode = 'r')
        y_train = np.load(path + '/task2/task2_y_train.npy', mmap_mode = 'r')
        X_test = np.load(path + '/task2/task2_X_test.npy', mmap_mode = 'r')
        y_test = np.load(path + '/task2/task2_y_test.npy', mmap_mode = 'r')
    elif task == 0:
        # Unlabeled images (both tasks)
        X_train = np.load(path + '/task0/task0_X_train.npy', mmap_mode = 'r')
        X_test = None
        y_train = None
        y_test = None
    return X_train, X_test, y_train, y_test
