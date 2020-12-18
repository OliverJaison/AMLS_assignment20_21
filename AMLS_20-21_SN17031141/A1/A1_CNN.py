#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import keras
import cv2
import pandas as pd
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import InputLayer
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import OneHotEncoder
import itertools
import matplotlib.pyplot as plt
import os
get_ipython().run_line_magic('matplotlib', 'inline')


# In[10]:


file_path = 'D:\Admin\Documents\Year_4\AMLS\Assessment\dataset_AMLS_20-21\celeba\img'
labels_path = 'D:\Admin\Documents\Year_4\AMLS\Assessment\dataset_AMLS_20-21\celeba\labels.csv'


# In[11]:


df = pd.read_csv(labels_path)

df["smiling"] = df["smiling"].replace(to_replace=[-1], value=['Frown'])
df["smiling"] = df["smiling"].replace(to_replace=[1], value=['Smile'])
df["gender"] = df["gender"].replace(to_replace=[-1], value=['Female'])
df["gender"] = df["gender"].replace(to_replace=[1], value=['Male'])

one_hot_s = pd.get_dummies(df["smiling"])
one_hot_g = pd.get_dummies(df["gender"])
df = df.drop(columns=["gender", "smiling"])
df = df.join(one_hot_g)
df = df.join(one_hot_s)


# In[12]:


train, validation, test =               np.split(df.sample(frac=1), 
                       [int(.6*len(df)), int(.8*len(df))])


# In[13]:


datagen = ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_dataframe(dataframe=train, 
                                              directory=file_path, 
                                              x_col="img_name", 
                                              y_col=["Female", "Male"], 
                                              class_mode="raw", 
                                              target_size=(55,45), 
                                              batch_size=30, 
                                              color_mode='grayscale', 
                                              interpolation='bicubic')

validation_generator = datagen.flow_from_dataframe(dataframe=validation, 
                                              directory=file_path, 
                                              x_col="img_name", 
                                              y_col=["Female", "Male"], 
                                              class_mode="raw", 
                                              target_size=(55,45), 
                                              batch_size=10, 
                                              color_mode='grayscale', 
                                              interpolation='bicubic')

test_generator = datagen.flow_from_dataframe(dataframe=test, 
                                              directory=file_path, 
                                              x_col="img_name", 
                                              y_col=["Female", "Male"], 
                                              class_mode="raw", 
                                              target_size=(55,45), 
                                              batch_size=10, 
                                              color_mode='grayscale', 
                                              interpolation='bicubic')


# In[14]:


model = Sequential()
model.add(InputLayer(input_shape=(55,45,1)))
model.add(Conv2D(filters=96, kernel_size=(4,4), strides=4, activation="relu", padding="same"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(filters=256, kernel_size=(3,3), strides=1, activation="relu", padding="same"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(filters=256, kernel_size=(2,2), activation="relu", strides=1, padding="same"))
model.add(Flatten())
model.add(Dense(units=512, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(units=512, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(2, activation="softmax"))


# In[15]:


model.summary()


# In[22]:


model.compile(Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])


# In[23]:


train_steps = train_generator.n//train_generator.batch_size
validation_steps = validation_generator.n//validation_generator.batch_size


# In[24]:


history = model.fit_generator(generator=train_generator, 
                    steps_per_epoch=train_steps,
                    validation_data=validation_generator,
                    validation_steps=validation_steps,
                    epochs=15)


# In[25]:


model.evaluate(test_generator)


# In[26]:


epochs = range(1, len(history.epoch) + 1)
plt.plot(epochs, history.history['loss'], label='Training Loss')
plt.plot(epochs, history.history['accuracy'], label='Training Accuracy')
plt.plot(epochs, history.history['val_loss'], label='Validation Loss')
plt.plot(epochs, history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss/Accuracy')
plt.legend()
plt.show()


# In[ ]:




