# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 22:21:29 2020

@author: viper
"""

# import libraries
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import GlobalAveragePooling2D, Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, RMSprop, Adam

from keras import backend as K
backend.set_image_data_format('channels_last')

#%%
#using 128*128 image matrix size 
img_rows = 128
img_cols = 128
num_channel = 1 # to set gray-scale
img_data_list = []

#load data images
PATH = os.getcwd()
print(PATH)
data_path = PATH + '/pokemon'
data_dir_list = os.listdir(data_path)

for dataset in data_dir_list:
    img_list = os.listdir(data_path + '/' + dataset)
    print('Loaded the images of dataset-' + '{}\n'.format(dataset))
    for img in img_list:
        input_img = cv2.imread(data_path + '/' + dataset + '/' + img)
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        input_img_resize = cv2.resize(input_img, (128, 128))
        img_data_list.append(input_img_resize)

#convert into array and output as grey scale for CNN model
img_data = np.array(img_data_list)
img_data = img_data.astype('float32')
img_data /= 255
print(img_data.shape) #250 samples, 128 rows 128 columns

#%%

# this 'if' statement is for reshaping gray-scale images
if num_channel == 1:
    if K.common.image_dim_ordering() == 'tf':
        img_data = np.expand_dims(img_data, axis=1)
        print(img_data.shape)
    else:
        img_data = np.expand_dims(img_data, axis=4)
        print(img_data.shape)
# this 'else' statement is for reshaping RGB images
else:
    if K.common.image_dim_ordering() == 'tf':
        img_data = np.rollaxis(img_data, 3, 1)
        print(img_data.shape)

#%%
#target labelling and one hot encoding of dataset        
num_classes = 5 #define number of class

num_of_samples = img_data.shape[0]
labels = np.ones((num_of_samples,), dtype='int64')

labels[0:49] = 0    # bulbsaur
labels[50:99] = 1  # charmander
labels[100:149] = 2  # mewtwo
labels[150:199] = 3  # pikachu
labels[200:250] = 4 # squirtle

names = ['bulbasaur', 'charmander', 'mewtwo', 'pikachu', 'squirtle']
Y = np_utils.to_categorical(labels, num_classes)

#%%
#setting up train test split data
x, y = shuffle(img_data, Y, random_state=2)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)
input_shape = img_data[0].shape
print(input_shape)

X_train = X_train.reshape(X_train.shape[0], img_cols, img_rows, 1)
X_test = X_test.reshape(X_test.shape[0], img_cols, img_rows, 1)
X_train.shape
#%%

#build the CNN model
model = Sequential()

model.add(Conv2D(32, (3, 3), padding="same", input_shape=(128,128,1)))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=-1))
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=1))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=-1))
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# softmax classifier
model.add(Dense(5))

# Last 'Dense' input is used to specify the number of classes to predict
model.add(Activation("softmax"))
model.compile(Adam(lr=0.0001), loss='categorical_crossentropy', metrics=["accuracy"])

model.summary()

#%%

hist = model.fit(X_train, y_train, batch_size=32, epochs=20,
                 verbose=1, validation_data=(X_test, y_test))
