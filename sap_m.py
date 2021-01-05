# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 13:14:31 2020

@author: safayet_khan
"""

import os
import glob
import csv
import numpy as np
import pandas as pd
import cv2
import matplotlib.image as mpimg
from skimage.util import random_noise


IMAGE_SIZE = 112
CHANNEL_NUMBER = 3
IMAGE_SIZE_3D = (IMAGE_SIZE, IMAGE_SIZE, CHANNEL_NUMBER)
IMAGE_SIZE_2D = (IMAGE_SIZE, IMAGE_SIZE)
MAIN_DIR = 'C:/Users/safayet_khan/Desktop/BCR/Numta/numta/'
FOLDER_LIST = [['training-a', '*.png'], ['training-b', '*.png'],
               ['training-c', '*.png'], ['training-d', '*.png'],
               ['training-e', '*.png'], ['training-f', '*.jpg'],
               ['training-g', '*.jpg'], ['training-h', '*.jpg'],
               ['training-i', '*.jpg'], ['training-j', '*.jpg'],
               ['training-l', '*.jpg']]
CSV_LIST = ['training-a.csv', 'training-b.csv', 'training-c.csv',
            'training-d.csv', 'training-e.csv', 'training-f-bit.csv',
            'training-g-color.csv', 'training-h-blackout.csv',
            'training-i-blend.csv', 'training-j-blur.csv',
            'training-l-bs.csv']
image_path = []
data_frame = pd.DataFrame()

for i in range(np.shape(FOLDER_LIST)[0]):
    temp_image_path = glob.glob(os.path.join(MAIN_DIR, FOLDER_LIST[i][0],
                                             FOLDER_LIST[i][1]))
    image_path += temp_image_path
print(np.shape(image_path))

for i in range(np.shape(CSV_LIST)[0]):
    temp_csv_path = os.path.join(MAIN_DIR, CSV_LIST[i])
    temp_data_frame = pd.read_csv(temp_csv_path)
    temp_data_frame = temp_data_frame[['filename', 'digit']]
    data_frame = data_frame.append(temp_data_frame)
data_frame = data_frame.set_index('filename')
print(np.shape(data_frame))


y_label = np.empty((np.shape(image_path)[0], 1), dtype=np.uint8)
x_array = np.empty((np.shape(image_path)[0], IMAGE_SIZE_3D[0],
                    IMAGE_SIZE_3D[1], IMAGE_SIZE_3D[2]), dtype=np.uint8)

for i, path in enumerate(image_path):
    name = path.split(sep=os.path.sep)[-1]
    y_label[i, ] = (data_frame.loc[name]['digit'])
    img_array = cv2.imread(path, cv2.IMREAD_COLOR)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    img_array = cv2.resize(src=img_array, dsize=IMAGE_SIZE_2D)
    x_array[i, ] = img_array
print(np.shape(x_array))
print(np.shape(y_label))


SAVE_INFO = ['training-m', 'training-m-sap.csv']
WRITE_PATH = os.path.join(MAIN_DIR, SAVE_INFO[0])
WRITE_CSV = os.path.join(MAIN_DIR, SAVE_INFO[1])
COUNTER = 0
NUM_NEW_IMAGE = 15000
name_label = [['filename', 'digit']]

if not os.path.exists(path=WRITE_PATH):
    os.mkdir(path=WRITE_PATH)
os.chdir(path=WRITE_PATH)

for img_array in range(0, NUM_NEW_IMAGE):
    name_label_row = []
    FILE_NAME = 'm{:05d}.jpg'.format(COUNTER)
    name_label_row.append(FILE_NAME)
    image_index = np.random.randint(0, x_array.shape[0])
    name_label_row.append(int(y_label[image_index]))
    name_label.append(name_label_row)

    noisy_image = random_noise(x_array[image_index], mode='s&p', clip=True)
    mpimg.imsave(FILE_NAME, noisy_image)
    COUNTER = COUNTER+1
print(np.shape(noisy_image))

with open(WRITE_CSV, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(name_label)
