# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 03:06:33 2020

@author: viper
"""

import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Activation
from keras.optimizers import SGD, RMSprop, Adam
from keras.models import Sequential
from keras.applications.vgg16 import  VGG16
from keras.preprocessing.image import ImageDataGenerator

#%%

#resizing images to this format for vgg16 cnn model
image_size = [224,224]

train_path = 'pokemon'

#for checking number of classes
folder = glob('pokemon/*')
len(folder)

#%%
vgg = VGG16(input_shape = image_size + [3], weights= 'imagenet', include_top=False)

#freezing all the layers of vgg16 EXCEPT the last 4 layers
for layer in vgg.layers[:-4]:
    layer.trainable= False
# Check the trainable status of the individual layers
for layer in vgg.layers:
    print(layer, layer.trainable)

#%%
#Tuning the pretrain model by adding a few layers
#appending last layer of model to specify number of classes output and using activation softmax
from keras import models
from keras import layers
from keras import optimizers

# Create the model
model = models.Sequential()

# Add the vgg convolutional base model
model.add(vgg)

# Add new layers
model.add(layers.Flatten())
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(len(folder), activation='softmax'))

# Show a summary of the model. Check the number of trainable parameters
model.summary()

#%%
#compiling the model
model.compile(RMSprop(lr=0.0001), loss= 'categorical_crossentropy', metrics= ['accuracy'])


#%%
#define the Image Generator and train / test set
train_datagen = ImageDataGenerator(rescale = 1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, validation_split=0.2)

train_set = train_datagen.flow_from_directory('pokemon',class_mode='categorical',
                                             target_size=(224,224),subset='training',
                                             batch_size=32)

test_set = train_datagen.flow_from_directory('pokemon',class_mode='categorical',
                                             target_size=(224,224),subset='validation',
                                             batch_size=32)

#%%
#training the model
hist= model.fit_generator(train_set, validation_data=test_set, epochs = 30, steps_per_epoch=len(train_set), validation_steps=len(test_set))
#achieving 99.5% training accuracy, validation acc 82%

#%%

#plotting the history training and validation loss to see if we are on the local minimum 
train_loss = hist.history['loss']
val_loss = hist.history['val_loss']
train_acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']
xc = range(30)

plt.figure(1, figsize=(7, 5))
plt.plot(xc, train_loss)
plt.plot(xc, val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train', 'val'])
plt.style.use(['classic'])

# saving the figure
plt.savefig('trial_psi_loss.png', bbox_inches='tight')

#as seen training loss reach almost local minimum although there is abit of flactuation in during 15-17 epocs
#val loss is lowest at around epoch 27 but fairly overfitting in the end of 30 epochs

#%%

plt.figure(2, figsize=(7, 5))
plt.plot(xc, train_acc)
plt.plot(xc, val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train', 'val'], loc=4)
plt.style.use(['classic'])

plt.savefig('trial_psi_acc.png', bbox_inches='tight')

#training accuracy is high enough, val acc could be higher by adding more layers or only un-freezing less than 4 last layers?
#Data augmentation on the last few layers could have yield better accuracy score but not really sure the method. 

#%%
import tensorflow as tf
import keras
image = tf.keras.preprocessing.image.load_img('mewtwo1.jpg', target_size=image_size)
input_arr = keras.preprocessing.image.img_to_array(image)
input_arr = np.array([input_arr])  # Convert single image to a batch.
predictions = model.predict(input_arr)
predictions
#%%
