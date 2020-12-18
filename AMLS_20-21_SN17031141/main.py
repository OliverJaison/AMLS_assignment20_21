# ======================================================================================================================
# Importing the necessary libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import os
import dlib
import keras
import matplotlib.image as mapimg
from sklearn.model_selection import train_test_split
from sklearn import svm
from keras.preprocessing import image
from sklearn import decomposition
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn import metrics
from skimage.color import rgb2gray
from skimage.feature import hog
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
# ======================================================================================================================
# Data preprocessing
# Getting the current working directory and moving into the Datasets folder from there
_dir = os.path.join(os.getcwd(), "Datasets")

# Simply getting the paths for the labels and the images for training and testing celeba
A_train_path = os.path.join(_dir, "celeba")
A_train_img_path = os.path.join(A_train_path, "img")
A_train_label_path = os.path.join(A_train_path, "labels.csv")
A_test_path = os.path.join(_dir, "celeba_test")
A_test_img_path = os.path.join(A_test_path, "img")
A_test_label_path = os.path.join(A_test_path, "labels.csv")

# Simply getting the paths for the labels and the images for training and testing cartoon set
B_train_path = os.path.join(_dir, "cartoon_set")
B_train_img_path = os.path.join(B_train_path, "img")
B_train_label_path = os.path.join(B_train_path, "labels.csv")
B_test_path = os.path.join(_dir, "cartoon_set_test")
B_test_img_path = os.path.join(B_test_path, "img")
B_test_label_path = os.path.join(B_test_path, "labels.csv")


# This function takes the shape of an array and returns the coordinates
def shape_to_np(shape, dtype="int"):
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


# This function takes a rectangle and returns the dimensions of it
def rect_to_dim(rect):
    w = rect.right() - rect.left()
    h = rect.top() - rect.bottom()
    return w, h


# This function takes in an image and uses the functions made above to find the coordinates of the 68 landmarks on
# the subject face in the image.
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


# This function iterates through every image in the dataset and generates a feature matrix from it that is indexed based
# on the image name.
def create_ld_matrix(file_path, sample_size, gender_labels, smile_labels):
    counter = 0
    features = []
    labels1 = []
    labels2 = []
    image_paths = [os.path.join(file_path, l) for l in os.listdir(file_path)]
    for img_path in image_paths:
        img = image.img_to_array(image.load_img(img_path, target_size=None, interpolation='bicubic'))
        file_name= img_path.split('\\')[-1]
        feature = create_feature(img)
        if feature is not None:
            features.append(feature)
            labels1.append(gender_labels[file_name])
            labels2.append(smile_labels[file_name])
            counter += 1
        if counter > sample_size - 1:
            break
    features = np.array(features)
    return features, labels1, labels2


# This function is used for B1 and it iterates through all images in the dataset and extracts the hog features from them
# and stores them in one big hog feature matrix.
def create_hog_matrix(file_path, train_df):
    features = []
    labels = []
    for file_name in train_df["file_names"]:
        img_path = os.path.join(file_path, file_name)
        img = cv2.imread(img_path, 0)
        feature = hog(img, pixels_per_cell=(6,6))
        if feature is not None:
            features.append(feature)
            temp = file_name.split(".")[0]
            labels.append(train_df.loc[int(temp), "face_shape"])
    features = np.array(features)
    return features, labels


# This function is for cropping out the right eye of a cartoon image from the Cartoon Set
def create_eye_matrix(file_path, df, starty, endy, startx, endx):
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


# Generating a dictionary for the gender labels and another for smile labels in celeba
labels_file = open(A_train_label_path, 'r')
lines = labels_file.readlines()
lines = lines[1:]
gender_label = {}
smile_label = {}
for line in lines:
    gender_label[line.split('\t')[1]] = line.split('\t')[2]
    smile_label[line.split('\t')[1]] = line.split('\t')[3]

# The same is done for the test labels
labels_file = open(A_test_label_path, 'r')
lines = labels_file.readlines()
lines = lines[1:]
gender_test_label = {}
smile_test_label = {}
for line in lines:
    gender_test_label[line.split('\t')[1]] = line.split('\t')[2]
    smile_test_label[line.split('\t')[1]] = line.split('\t')[3]

# Loads the training labels for B into a dataframe
train_df = pd.read_csv(B_train_label_path)
# Specifies how many of the images in the training dataset will be used
sample_size = 1000
# Here the dataframe is cleaned up a bit just to make debugging easier if something goes wrong
train_df = train_df.drop(columns="Unnamed: 0")
temp = train_df["file_name"]
train_df.insert(loc=0, column="file_names", value=temp)
train_df = train_df.drop(columns="file_name")
# Here the class system of 0-4 is converted to labels
train_df["eye_color"] = train_df["eye_color"].replace(to_replace=[0], value=["Brown"])
train_df["eye_color"] = train_df["eye_color"].replace(to_replace=[1], value=["Blue"])
train_df["eye_color"] = train_df["eye_color"].replace(to_replace=[2], value=["Green"])
train_df["eye_color"] = train_df["eye_color"].replace(to_replace=[3], value=["Hazel"])
train_df["eye_color"] = train_df["eye_color"].replace(to_replace=[4], value=["Black"])
# Here the labelling system is now converted to one hot encoding
one_hot = pd.get_dummies(train_df["eye_color"])
# Here the one hot encoding is added to the data frame.
train_df = train_df.join(one_hot)
# Here the dataframe drops off all except the first (sample_size) units in order to save on processing time.
train_df = train_df.drop(train_df.index[sample_size:])
# The resulting dataframe will have the labels for face shape as 0-4 and labels for eye colour in one hot encoding

# The same is done for the test labels
test_df = pd.read_csv(B_test_label_path)
# Specifies how many of the images in the training dataset will be used
sample_size = 2500
# Here the dataframe is cleaned up a bit just to make debugging easier if something goes wrong
test_df = test_df.drop(columns="Unnamed: 0")
temp = test_df["file_name"]
test_df.insert(loc=0, column="file_names", value=temp)
test_df = test_df.drop(columns="file_name")
# Here the class system of 0-4 is converted to labels
test_df["eye_color"] = test_df["eye_color"].replace(to_replace=[0], value=["Brown"])
test_df["eye_color"] = test_df["eye_color"].replace(to_replace=[1], value=["Blue"])
test_df["eye_color"] = test_df["eye_color"].replace(to_replace=[2], value=["Green"])
test_df["eye_color"] = test_df["eye_color"].replace(to_replace=[3], value=["Hazel"])
test_df["eye_color"] = test_df["eye_color"].replace(to_replace=[4], value=["Black"])
# Here the labelling system is now converted to one hot encoding
one_hot = pd.get_dummies(test_df["eye_color"])
# Here the one hot encoding is added to the data frame.
test_df = test_df.join(one_hot)
# Here the dataframe drops off all except the first (sample_size) units in order to save on processing time.
test_df = test_df.drop(test_df.index[sample_size:])
# The resulting dataframe will have the labels for face shape as 0-4 and labels for eye colour in one hot encoding

# Now we call the functions to generate the features and the labels
A_train_f, A_train_l1, A_train_l2 = create_ld_matrix(file_path=A_train_img_path,
                                                     sample_size=5000,
                                                     gender_labels=gender_label,
                                                     smile_labels=smile_label)
A_train_l1 = (np.array(A_train_l1).astype(int)+1)/2
A_train_l1 = A_train_l1.astype(int)
A_train_l2 = (np.array(A_train_l2).astype(int)+1)/2
A_train_l2 = A_train_l2.astype(int)
A_train_f = A_train_f.reshape((A_train_f//136, 136))

# Here we split the training data into training and testing.
A1_train_x, A1_test_x, A1_train_y, A1_test_y = train_test_split(A_train_f, A_train_l1, test_size=0.25)
A2_train_x, A2_test_x, A2_train_y, A2_test_y = train_test_split(A_train_f, A_train_l2, test_size=0.25)

# We do the same for the test dataset
A_test_f, A_test_l1, A_test_l2 = create_ld_matrix(file_path=A_test_img_path,
                                                  sample_size=1000,
                                                  gender_labels=gender_test_label,
                                                  smile_labels=smile_test_label)
A_test_l1 = (np.array(A_test_l1).astype(int)+1)/2
A_test_l1 = A_test_l1.astype(int)
A_test_l2 = (np.array(A_test_l2).astype(int)+1)/2
A_test_l2 = A_test_l2.astype(int)
A_test_f = A_test_f.reshape((A_test_f//136, 136))

# Now we do the same for the Cartoon Set
B1_train_df, B1_val_df, B1_test_df = \
              np.split(train_df.sample(frac=1),
                       [int(.6*len(train_df)), int(.8*len(train_df))])

B1_train_f, B1_train_l = create_hog_matrix(B_train_img_path, B1_train_df)
B1_val_f, B1_val_l = create_hog_matrix(B_train_img_path, B1_val_df)
B1_test_f, B1_test_l = create_hog_matrix(B_train_img_path, B1_test_df)
B1_mark_f, B1_mark_l = create_hog_matrix(B_test_img_path, test_df)

# Here we split the dataframe into training, validation and test with a ratio of 6:2:2
B2_train_df, B2_validation_df, B2_test_df = \
              np.split(train_df.sample(frac=1),
                       [int(.6*len(train_df)), int(.8*len(train_df))])

# These are used for the Haar face and eye cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# These are the coordinates to crop out the right eye of every cartoon image.
# The method to do this is shown in the B2_CNN file and outlined in the report
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

# The values from above are used as parameters for generating a matrix for the region of interest in the image and the
# labels
B2_train_eyes, B2_train_labels = create_eye_matrix(B_train_img_path, B2_train_df, starty, endy, startx, endx)
B2_val_eyes, B2_val_labels = create_eye_matrix(B_train_img_path, B2_validation_df, starty, endy, startx, endx)
B2_test_eyes, B2_test_labels = create_eye_matrix(B_train_img_path, B2_test_df, starty, endy, startx, endx)

# Here the same thing is done for the test dataset and the test dataframe
B2_mark_eyes, B2_mark_labels = create_eye_matrix(B_test_img_path, test_df, starty, endy, startx, endx)

# Here we instantiate the ImageDataGenerator Class
datagen = ImageDataGenerator()

# Now we make data flow from these arrays in batches to the model using this class
# The default batch size is 32.
train_generator = datagen.flow(x=B2_train_eyes, y=B2_train_labels)

validation_generator = datagen.flow(x=B2_val_eyes, y=B2_val_labels)

test_generator = datagen.flow(x=B2_test_eyes, y=B2_test_labels)

mark_generator = datagen.flow(x=B2_mark_eyes, y=B2_mark_labels)
# ======================================================================================================================
# Task A1
# Model_A1 is an SVM with polynomial kernel of degree 3 and hyperparameter 1
model_A1 = svm.SVC(kernel='poly', degree=3, C=1.0)
# Model_A1 is fit to the training data and labels
model_A1.fit(A1_train_x, A1_train_y)
# The accuracy is now seen for both the validation and test sets.
acc_A1_train = metrics.accuracy_score(A1_test_y, model_A1.predict(A1_test_x))
acc_A1_test = metrics.accuracy_score(A_test_l1, model_A1.predict(A_test_f))

# ======================================================================================================================
# Task A2
model_A2 = svm.SVC(kernel='poly', degree=3, C=1.0)
model_A2.fit(A2_train_x, A2_train_y)
acc_A2_train = metrics.accuracy_score(A2_test_y, model_A2.predict(A2_test_x))
acc_A2_test = metrics.accuracy_score(A_test_l2, model_A2.predict(A_test_f))

# ======================================================================================================================
# Task B1
model_B1 = svm.SVC(kernel='poly', degree=3, C=1.0)
model_B1.fit(B1_train_f, B1_train_l)
acc_B1_train = metrics.accuracy_score(B1_val_l, model_B1.predict(B1_val_f))
acc_B1_test = metrics.accuracy_score(B1_mark_l, model_B1.predict(B1_mark_f))

# ======================================================================================================================
# Task B2
# Here we build the CNN model
model_B2 = Sequential()
model_B2.add(InputLayer(input_shape=train_eyes[0].shape))
model_B2.add(Conv2D(filters=96, kernel_size=(2, 2), strides=4, activation="relu", padding="same"))
model_B2.add(MaxPooling2D(pool_size=(2, 2)))
model_B2.add(BatchNormalization())
model_B2.add(Flatten())
model_B2.add(Dense(units=512, activation="relu"))
model_B2.add(Dropout(0.2))
model_B2.add(Dense(units=512, activation="relu"))
model_B2.add(Dropout(0.2))
model_B2.add(Dense(5, activation="softmax"))

# Here we compile using Adam optimizer with binary crossentropy loss and a learning rate of 0.0001
model_B2.compile(Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

train_steps = train_generator.n//train_generator.batch_size
validation_steps = validation_generator.n//validation_generator.batch_size

# The model is now fit to the training data
history = model_B2.fit(x=train_generator,
          steps_per_epoch=train_steps,
          validation_data=validation_generator,
          validation_steps=validation_steps,
          epochs=10)

# Here we can see the performance of the model with the validation and test data
acc_B2_train = model_B2.evaluate(test_generator)[1]
acc_B2_test = model_B2.evaluate(mark_generator)[1]

# ======================================================================================================================
# Print out your results with following format:
print('TA1:{},{};TA2:{},{};TB1:{},{};TB2:{},{};'.format(acc_A1_train, acc_A1_test,
                                                        acc_A2_train, acc_A2_test,
                                                        acc_B1_train, acc_B1_test,
                                                        acc_B2_train, acc_B2_test))
