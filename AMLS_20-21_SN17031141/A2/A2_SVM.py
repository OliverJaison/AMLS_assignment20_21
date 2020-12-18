#!/usr/bin/env python
# coding: utf-8

# Importing the necessary libraries

# In[1]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import os
import dlib
from sklearn.model_selection import train_test_split
from sklearn import svm
from keras.preprocessing import image
from sklearn import decomposition
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn import metrics
from skimage.color import rgb2gray
import pickle


# Initialising the file path and the sample size to test

# In[2]:


file_path = 'D:\Admin\Documents\Year_4\AMLS\Assessment\dataset_AMLS_20-21\celeba\img'
sample_size = 5000


# Generating a dictionary for the labels against the file names

# In[3]:


labels_file = open('D:\Admin\Documents\Year_4\AMLS\Assessment\dataset_AMLS_20-21\celeba\labels.csv', 'r')
lines = labels_file.readlines()
lines = lines[1:]
smile_label = {}
for line in lines:
    smile_label[line.split('\t')[1]] = line.split('\t')[3]


# In[4]:


def shape_to_np(shape, dtype="int"):
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


# In[5]:


def rect_to_dim(rect):
    w = rect.right() - rect.left()
    h = rect.top() - rect.bottom()
    return (w, h)


# In[6]:


def create_feature(img):
    face_detect = dlib.get_frontal_face_detector()
    shape_predict = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = gray.astype('uint8')
    rects = face_detect(gray, 1)
    num_faces = len(rects)
    
    if num_faces == 0:
        return None

    face_areas = np.zeros((1, num_faces))
    face_shapes = np.zeros((136, num_faces), dtype=np.int64)
    
    for (i, rect) in enumerate(rects):
        temp_shape = shape_predict(gray, rect)
        temp_shape = shape_to_np(temp_shape)
        (w, h) = rect_to_dim(rect)
        face_shapes[:, i] = np.reshape(temp_shape, [136])
        face_areas[0, i] = w * h
        dlibout = np.reshape(np.transpose(face_shapes[:, np.argmax(face_areas)]), [68, 2])
    return dlibout


# In[7]:


def create_feature_matrix(file_path, sample_size, gender_labels):
    counter = 0
    features = []
    labels = []
    image_paths = [os.path.join(file_path, l) for l in os.listdir(file_path)]
    for img_path in image_paths:
        img = image.img_to_array(image.load_img(img_path, target_size=None, interpolation='bicubic'))
        file_name= img_path.split('\\')[-1]
        feature = create_feature(img)
        if feature is not None:
            features.append(feature)
            labels.append(gender_labels[file_name])
            counter += 1
        if counter > sample_size - 1:
            break
    features = np.array(features)
    return features, labels


# In[8]:


x, y = create_feature_matrix(file_path, sample_size, smile_label)


# Formatting the label array and the feature array

# In[9]:


y = (np.array(y).astype(int) + 1)/2
y = y.astype(int)


# In[10]:


x = x.reshape((x.size//136, 68*2))


# Splitting the input data into training and testing

# In[11]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)


# Initialising the classifier

# In[12]:


classifier = svm.SVC(kernel='poly', degree=3, C=1.0)


# In[13]:


classifier.fit(x_train, y_train)


# In[14]:


y_pred = classifier.predict(x_test)
accuracy = metrics.accuracy_score(y_test,y_pred=y_pred)
print(accuracy)


# In[15]:


labels_file = open('D:\Admin\Documents\Year_4\AMLS\Assessment\dataset_AMLS_20-21\celeba_test\labels.csv', 'r')
lines = labels_file.readlines()
lines = lines[1:]
test_label = {}
for line in lines:
    test_label[line.split('\t')[1]] = line.split('\t')[3]


# In[16]:


test_path = 'D:\Admin\Documents\Year_4\AMLS\Assessment\dataset_AMLS_20-21\celeba_test\img'


# In[17]:


test_x, test_y = create_feature_matrix(test_path, 1000, test_label)


# In[18]:


test_y = (np.array(test_y).astype(int) + 1)/2
test_y = test_y.astype(int)
test_x = test_x.reshape((test_x.size//136, 136))


# In[19]:


test_pred = classifier.predict(test_x)
test_accuracy = metrics.accuracy_score(test_y, y_pred=test_pred)
print(test_accuracy)


# In[20]:


model_name = "Raw_SVM.sav"


# In[21]:


pickle.dump(classifier, open(model_name, "wb"))


# In[ ]:




