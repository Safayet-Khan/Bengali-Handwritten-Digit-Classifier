# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 17:22:02 2021

@author: safayet_khan
"""

# This script demonstrates training a Convolutional Neural Network (CNN)
# to classify Bengali Handwritten Digits NumtaDB
# (https://www.kaggle.com/BengaliAI/numta/) images. The CNN architecture
# used in this notebook is similar to the VGG16
# (https://arxiv.org/pdf/1409.1556.pdf) but the model used in
# this notebook has BatchNormalization layers.


# Importing necessary libraries
import math
import os
import glob
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import CSVLogger, TerminateOnNaN
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
tf.config.experimental_run_functions_eagerly(True)


# Rescaling and creating augmentation with ImageDataGenerator
datagen = ImageDataGenerator(rotation_range=45,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             brightness_range=[0.7, 1.3],
                             zoom_range=[1.0, 1.5],
                             rescale=1/255.0,
                             shear_range=30,
                             fill_mode='constant',
                             cval=0)


# Image size, batch size, and other necessary values are being fixed
RANDOM_SEED = 42
BATCH_SIZE = 128
IMAGE_SIZE = 112
CHANNEL_NUMBER = 3
IMAGE_SIZE_2D = (IMAGE_SIZE, IMAGE_SIZE)
IMAGE_SIZE_3D = (IMAGE_SIZE, IMAGE_SIZE, CHANNEL_NUMBER)
CLASSES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
TRAIN_DIR = 'C:/Users/safayet_khan/Desktop/BCR/Numta/numta_flow/train/'
VAL_DIR = 'C:/Users/safayet_khan/Desktop/BCR/Numta/numta_flow/val/'


# Data generator for the Training set
train_generator = datagen.flow_from_directory(directory=TRAIN_DIR,
                                              target_size=IMAGE_SIZE_2D,
                                              color_mode='rgb',
                                              classes=CLASSES,
                                              class_mode='categorical',
                                              batch_size=BATCH_SIZE,
                                              shuffle=True,
                                              seed=RANDOM_SEED)
print(train_generator.class_indices)


# Data generator for the Validation set
validation_generator = datagen.flow_from_directory(directory=VAL_DIR,
                                                   target_size=IMAGE_SIZE_2D,
                                                   color_mode='rgb',
                                                   classes=CLASSES,
                                                   class_mode='categorical',
                                                   batch_size=BATCH_SIZE,
                                                   shuffle=True,
                                                   seed=RANDOM_SEED)
print(validation_generator.class_indices)


# Creating the Model and printing model summary
model = Sequential()
model.add(Input(shape=IMAGE_SIZE_3D))
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same',
                  activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same',
                  activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))

model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same',
                  activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same',
                  activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))

model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same',
                  activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same',
                  activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))

model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same',
                  activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same',
                  activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same',
                  activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))

model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same',
                  activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same',
                  activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same',
                  activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))

model.add(Flatten())
model.add(Dense(units=1024, activation='relu'))
model.add(Dropout(rate=0.3))
model.add(Dense(units=1024, activation='relu'))
model.add(Dropout(rate=0.3))
model.add(Dense(units=10, activation='softmax'))
model.summary()


# Callback (Stopped after reaching a certain value)
class MyCallback(Callback):
    '''
    Stop training after val_accuracy reached a certain number
    '''
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('val_accuracy')>0.9999:
            print('\nCOMPLETED!!!')
            self.model.stop_training=True
callbacks = MyCallback()


# ModelCheckpoint Callback
FOLDER_PATH = 'C:/Users/safayet_khan/Desktop/code-final/checkpoint/'
if not os.path.exists(path=FOLDER_PATH):
    os.mkdir(path=FOLDER_PATH)

CHECKPOINT_FILEPATH = os.path.join(FOLDER_PATH, 'model.h5')
checkpoint_callback = ModelCheckpoint(filepath=CHECKPOINT_FILEPATH,
                                      monitor='val_loss', mode='min',
                                      verbose=1, save_weights_only=False,
                                      save_best_only=True, save_freq='epoch')


# LearningRateScheduler Callback
EPSILON = 0.1
INITIAL_LEARNING_RATE = 0.001

def lr_step_decay(epoch, learning_rate=INITIAL_LEARNING_RATE):
    '''
    Reduce the learning rate by a certain percentage after a certain
    number of epochs.
    '''
    drop_rate = 0.25
    epochs_drop = 5.0
    return INITIAL_LEARNING_RATE * math.pow((1-drop_rate),
                                            math.floor(epoch/epochs_drop))

lr_callback = LearningRateScheduler(schedule=lr_step_decay, verbose=1)


# CSVLogger and TerminateOnNaN Callback
LOG_FILEPATH = 'C:/Users/safayet_khan/Desktop/code-final/log.csv'
CSV_callback = CSVLogger(filename=LOG_FILEPATH, separator=',', append=False)
Terminate_callback = TerminateOnNaN()


# Step per epoch, Validation steps, and number of Epochs to be trained
STEPS_PER_EPOCH = math.ceil(train_generator.samples/BATCH_SIZE)
VALIDATION_STEPS = math.ceil(validation_generator.samples/BATCH_SIZE)
EPOCHS = 50


# Compiling the Model
model.compile(optimizer=Adam(learning_rate=INITIAL_LEARNING_RATE,
                             epsilon=EPSILON),
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# Training the Model
model.fit(train_generator, shuffle=True, epochs=EPOCHS,
          steps_per_epoch=STEPS_PER_EPOCH, verbose=1,
          validation_data=validation_generator,
          validation_steps=VALIDATION_STEPS,
          callbacks=[callbacks, checkpoint_callback, lr_callback,
                     CSV_callback, Terminate_callback])


# Loading Test images paths
TEST_DIR = 'C:/Users/safayet_khan/Desktop/BCR/Numta/numta_flow/test/'
FOLDER_LIST = [['testing-a', '*.png'], ['testing-b', '*.png'],
               ['testing-c', '*.png'], ['testing-d', '*.png'],
               ['testing-e', '*.png'], ['testing-f', '*.png'],
               ['testing-f', '*.JPG'], ['testing-auga', '*.png'],
               ['testing-augc', '*.png']]
test_image_paths = []

for i in range(np.shape(FOLDER_LIST)[0]):
    temp_image_paths = glob.glob(os.path.join(TEST_DIR, FOLDER_LIST[i][0],
                                              FOLDER_LIST[i][1]))
    test_image_paths += temp_image_paths

print(np.shape(test_image_paths))


# Loading the Test images in an array. Resizing and Rescaling is also done
x_test = np.empty((np.shape(test_image_paths)[0], IMAGE_SIZE_3D[0],
                   IMAGE_SIZE_3D[1], IMAGE_SIZE_3D[2]), dtype=np.uint8)
test_file_names = []

for i, file_path in enumerate(test_image_paths):
    file_name = file_path.split(sep=os.path.sep)[-1]
    test_file_names.append(file_name)
    img_array = cv2.imread(file_path, cv2.IMREAD_COLOR)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB )
    img_array = cv2.resize(src=img_array, dsize=IMAGE_SIZE_2D)
    x_test[i, ] = img_array
x_test = x_test/255.0

print(type(x_test))
print(x_test.shape)


# Visualizing a test image
plt.imshow(np.reshape(x_test[20], IMAGE_SIZE_3D), cmap='gray')
plt.show()


# Loading the best Model
best_model = load_model(CHECKPOINT_FILEPATH)


# Making a prediction with it
prediction_probabilities = best_model.predict(x_test)

print(type(prediction_probabilities))
print(np.shape(prediction_probabilities))
print(prediction_probabilities[5])


# Extracting predicted labels from the prediction
prediction_labels = np.argmax(prediction_probabilities, axis=1)

print(type(prediction_labels))
print(np.shape(prediction_labels))
print(prediction_labels[5])


# Converting  the results to a CSV file
RESULT_FILEPATH = 'C:/Users/safayet_khan/Desktop/code-final/numta.csv'
result = {'key': test_file_names, 'label': prediction_labels}
result = pd.DataFrame(result)
result.to_csv(RESULT_FILEPATH, index=False)
