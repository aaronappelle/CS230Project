#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 22:00:40 2021

@author: Aaron
"""

from dataloader import load_task

path = '/Users/Aaron/Google Drive/Documents/Stanford University/Winter 2021/CS 230/CS230 Project/Data'

Xtrain1, Xtest1, ytrain1, ytest1 = load_task(1,path)
Xtrain2, Xtest2, ytrain2, ytest2 = load_task(2,path)

