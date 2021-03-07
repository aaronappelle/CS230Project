#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 23:06:22 2021

@author: Aaron

Create dataset of unlabeled images

"""

path = '/Users/Aaron/Google Drive/Documents/Stanford University/Winter 2021/CS 230/CS230 Project/Data/unlabeled'

# from tensorflow.keras.preprocessing import image_dataset_from_directory
# from keras.preprocessing.image import load_img, img_to_array

# unlabeled = image_dataset_from_directory(
#     path,
#     labels="inferred",
#     label_mode=None,
#     class_names=None,
#     color_mode="rgb",
#     batch_size=32,
#     image_size=(224, 224),
#     shuffle=True,
#     seed=None,
#     validation_split=None,
#     subset=None,
#     interpolation="bilinear",
#     follow_links=False,
# )

# # unlabeled_np = unlabeled.as_numpy()

from PIL import Image
import glob
import numpy as np

image_list = []
for filename in glob.glob(path + '/*.jpg'): #assuming gif
    im = Image.open(filename)
    im = im.resize((224,224))
    image_list.append(im)

images_np = np.array([np.array(im)[:,:,::-1] for im in image_list]) # transform to bgr to match training set

np.save('task0_X_train.npy',images_np)
    
#%%

r = np.random.randint(0,len(image_list))
r_im = image_list[r]
r_im.show()
r_im_np = np.array(r_im)
r_im2 = Image.fromarray(r_im_np)
r_im2.show()


