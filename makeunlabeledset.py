#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 23:06:22 2021

@author: Aaron

Create dataset of unlabeled images

"""

# path = '/Users/Aaron/Google Drive/Documents/Stanford University/Winter 2021/CS 230/CS230 Project/Data/unlabeled'
path = '/Users/Aaron/Google Drive/Documents/Stanford University/Winter 2021/CS 230/CS230 Project/Data/Unlabeled images'

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

#%%
# Shuffle and shorten (for speed, for now)
np.random.seed(10)
indices = np.arange(0, len(images_np))
test_indices = np.random.randint(0,len(images_np),(200,))
train_indices = np.delete(indices, test_indices)
X_test_unlabeled = images_np.take(test_indices,axis=0)
X_train_unlabeled = images_np.take(train_indices,axis=0)


#%% Save

np.save('/Users/Aaron/Google Drive/Documents/Stanford University/Winter 2021/CS 230/CS230 Project/Data/task0/task0_X_train.npy',X_train_unlabeled)
np.save('/Users/Aaron/Google Drive/Documents/Stanford University/Winter 2021/CS 230/CS230 Project/Data/task0/task0_X_test.npy',X_test_unlabeled)

# np.save('task0_X_train.npy',images_np)
# np.save('/Users/Aaron/Google Drive/Documents/Stanford University/Winter 2021/CS 230/CS230 Project/Data/task0/task0_X_train.npy',images_np)

#%% Label test set

from keras.utils import to_categorical
# from keras.preprocessing.image import array_to_img
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

X_test_unlabeled = np.load('/Users/Aaron/Google Drive/Documents/Stanford University/Winter 2021/CS 230/CS230 Project/Data/task0/task0_X_test.npy')

task1labels= [];
task2labels= [];

for im_array in X_test_unlabeled:
    # accept = False
    # while not accept:
    #     imshow(im_array[:,:,::-1]) # images in bgr
    #     plt.show()
    #     task1lab = int(input('0 (Object), 1 (Pixel), 2(Structure): '))
    #     task2lab = int(input('0 (Damaged), 1 (Undamaged): '))
    #     # im.close()
    #     plt.close()
    #     task1labels.append(task1lab)
    #     task2labels.append(task2lab)
    #     accept = bool(input('Accept Labels? (0/1):'))

    imshow(im_array[:,:,::-1]) # images in bgr
    plt.show()
    plt.close()
    
#%%
    
# task1labels = [2,0,2,0,1,2,2,2,2,2,0,0,0,2,0,2,0,2,2,2,2,2,2,2,0,0,2,2,2,2,1,2,2,2,2,2,2,0,2,2,2,2,0,2,2,2,2,2,2,2]
# task2labels = [1,0,1,1,0,1,0,1,0,1,1,1,0,0,0,1,1,1,0,0,1,0,1,1,0,0,0,1,1,1,0,0,1,1,1,0,1,0,1,1,0,0,0,0,1,1,0,1,0,0]
    
task1labels = [2,0,2,0,1,2,2,2,2,2,0,0,0,2,0,2,0,2,2,2,2,2,2,2,0,0,2,2,2,2,1,
               2,2,2,2,2,2,0,2,2,2,2,0,2,2,2,2,2,2,2,0,2,2,2,0,0,2,0,1,0,2,2,
               2,0,0,0,1,0,0,2,0,2,2,0,2,0,2,0,2,2,0,2,0,2,0,0,0,0,2,2,0,0,2,
               0,2,0,0,0,0,2,0,0,2,0,2,0,0,2,2,0,0,1,0,0,2,0,0,2,0,0,0,0,0,0,
               0,2,0,0,2,0,2,0,2,0,2,0,0,2,0,2,0,2,0,0,2,2,1,2,0,1,0,2,0,2,0,
               0,2,2,2,2,0,2,0,2,2,0,2,2,0,2,1,2,2,2,1,0,2,0,0,0,0,2,2,2,0,2,
               0,2,0,0,2,0,2,2,0,0,2,0,0,0]

task2labels = [1,0,1,1,0,1,0,1,0,1,1,1,0,0,0,1,1,1,0,0,1,0,1,1,0,0,0,1,1,1,0,
               0,1,1,1,0,1,0,1,1,0,0,0,0,1,1,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,
               1,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,1,0,1,1,0,1,0,1,0,0,1,0,0,0,0,
               0,1,1,0,0,0,1,1,0,1,1,1,0,1,1,1,0,1,1,0,0,0,0,0,1,0,0,0,0,0,1,
               0,1,0,1,1,0,1,0,0,1,1,1,1,0,0,0,1,0,1,0,1,0,0,1,0,0,0,1,1,1,1,
               1,0,1,0,0,0,0,1,1,0,1,0,0,1,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,
               1,1,0,0,1,1,0,0,0,0,1,0,1,0]

y_test_unlabeled_1 = to_categorical(np.array(task1labels))
y_test_unlabeled_2 = to_categorical(np.array(task2labels))

np.save('/Users/Aaron/Google Drive/Documents/Stanford University/Winter 2021/CS 230/CS230 Project/Data/task0/y_test_unlabeled_1.npy',y_test_unlabeled_1)
np.save('/Users/Aaron/Google Drive/Documents/Stanford University/Winter 2021/CS 230/CS230 Project/Data/task0/y_test_unlabeled_2.npy',y_test_unlabeled_2)
    
#%% Test image regeneration from array

r = np.random.randint(0,len(image_list))
r_im = image_list[r]
r_im.show()
r_im_np = np.array(r_im)
r_im2 = Image.fromarray(r_im_np)
r_im2.show()


