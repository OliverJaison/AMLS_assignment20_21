#!/usr/bin/env python
# coding: utf-8

# 07391449,
# 
# 08436710

# In[1]:


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


# In[2]:


file_path = "D:\Admin\Documents\Year_4\AMLS\Assessment\dataset_AMLS_20-21\cartoon_set\img"
labels_path = "D:\Admin\Documents\Year_4\AMLS\Assessment\dataset_AMLS_20-21\cartoon_set\labels.csv"


# In[3]:


df = pd.read_csv(labels_path)
sample_size = 1000
df = df.drop(columns="Unnamed: 0")

temp = df["file_name"]
df.insert(loc=0, column="file_names", value=temp)
df = df.drop(columns="file_name")

df["eye_color"] = df["eye_color"].replace(to_replace=[0], value=["Brown"])
df["eye_color"] = df["eye_color"].replace(to_replace=[1], value=["Blue"])
df["eye_color"] = df["eye_color"].replace(to_replace=[2], value=["Green"])
df["eye_color"] = df["eye_color"].replace(to_replace=[3], value=["Hazel"])
df["eye_color"] = df["eye_color"].replace(to_replace=[4], value=["Black"])

one_hot = pd.get_dummies(df["eye_color"])
df = df.join(one_hot)

df = df.drop(df.index[sample_size:])


# In[4]:


train, validation, test =               np.split(df.sample(frac=1), 
                       [int(.6*len(df)), int(.8*len(df))])


# In[5]:


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')


# In[7]:


y = 149
x = 128
ry = 83
rx = 48
rw = 56
rh = 56
starty = y + ry
endy = starty + rh
startx = x + rx
endx = startx + rw


# In[8]:


def create_eye_matrix(file_path, df, face, eye, starty, endy, startx, endx):
    eyes = []
    labels = []
    for file_name in df["file_names"]:
        img_path = os.path.join(file_path, file_name)
        img = cv2.imread(img_path)
        crop = img[starty:endy, startx:endx]
        eyes.append(crop)
        temp = file_name.split(".")[0]
        labels.append([df["Black"].loc[int(temp)], 
                       df["Blue"].loc[int(temp)], 
                       df["Brown"].loc[int(temp)], 
                       df["Green"].loc[int(temp)], 
                       df["Hazel"].loc[int(temp)]])
    eyes = np.array(eyes).astype(int)
    return eyes, labels


# In[9]:


train_eyes, train_labels = create_eye_matrix(file_path, train, face_cascade, eye_cascade, starty, endy, startx, endx)


# In[10]:


val_eyes, val_labels = create_eye_matrix(file_path, validation, face_cascade, eye_cascade, starty, endy, startx, endx)


# In[11]:


test_eyes, test_labels = create_eye_matrix(file_path, test, face_cascade, eye_cascade, starty, endy, startx, endx)


# In[12]:


datagen = ImageDataGenerator()

train_generator = datagen.flow(x = train_eyes, y = train_labels)

validation_generator = datagen.flow(x = val_eyes, y = val_labels)

test_generator = datagen.flow(x = test_eyes, y = test_labels)


# In[13]:


model = Sequential()
model.add(InputLayer(input_shape=train_eyes[0].shape))
model.add(Conv2D(filters=96, kernel_size=(2,2), strides=4, activation="relu", padding="same"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(units=512, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(units=512, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(5, activation="softmax"))


# In[14]:


model.summary()


# In[15]:


model.compile(Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])


# In[16]:


train_steps = train_generator.n//train_generator.batch_size
validation_steps = validation_generator.n//validation_generator.batch_size


# In[17]:


history = model.fit(x=train_generator, 
          steps_per_epoch=train_steps,
          validation_data=validation_generator,
          validation_steps=validation_steps,
          epochs=25)


# In[18]:


epochs = range(1, len(history.epoch) + 1)
plt.plot(epochs, history.history['loss'], label='Training Loss')
plt.plot(epochs, history.history['accuracy'], label='Training Accuracy')
plt.plot(epochs, history.history['val_loss'], label='Validation Loss')
plt.plot(epochs, history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss/Accuracy')
plt.legend()
plt.show()


# In[19]:


model.evaluate(x=test_generator)


# In[20]:


test_file_path = "D:\Admin\Documents\Year_4\AMLS\Assessment\dataset_AMLS_20-21\cartoon_set_test\img"
test_labels_path = "D:\Admin\Documents\Year_4\AMLS\Assessment\dataset_AMLS_20-21\cartoon_set_test\labels.csv"


# In[27]:


test_df = pd.read_csv(test_labels_path, sep="\t")
temp = test_df["file_name"]
test_df.insert(loc=0, column="file_names", value=temp)
test_df = test_df.drop(columns="file_name")
test_df = test_df.drop(columns="Unnamed: 0")

test_df["eye_color"] = test_df["eye_color"].replace(to_replace=[0], value=["Brown"])
test_df["eye_color"] = test_df["eye_color"].replace(to_replace=[1], value=["Blue"])
test_df["eye_color"] = test_df["eye_color"].replace(to_replace=[2], value=["Green"])
test_df["eye_color"] = test_df["eye_color"].replace(to_replace=[3], value=["Hazel"])
test_df["eye_color"] = test_df["eye_color"].replace(to_replace=[4], value=["Black"])

one_hot = pd.get_dummies(test_df["eye_color"])
test_df = test_df.join(one_hot)
print(test_df.head())


# In[28]:


test_img, test_labels = create_eye_matrix(file_path, train, face_cascade, eye_cascade, starty, endy, startx, endx)


# In[29]:


test_flow = datagen.flow(x = test_img, y = test_labels)


# In[30]:


model.evaluate(x=test_flow)


# In[31]:


img_path = os.path.join(file_path, df["file_names"].iloc[3])
img = cv2.imread(img_path)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray)
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        roi_color_eye = roi_color[ey:ey+eh, ex:ex+ew]
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        roi = roi_color[ey:ey+eh, ex: ex+ew]
plt.subplot(1,2,1)
plt.imshow(img)
plt.subplot(1,2,2)
plt.imshow(roi)


# In[ ]:




