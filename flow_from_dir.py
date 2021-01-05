# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 18:21:54 2020

@author: safayet_khan
"""

import os
import shutil
import pandas as pd
import numpy as np


MAIN_DIR = 'C:/Users/safayet_khan/Desktop/BCR/Numta/numta/'
CSV_LIST = ['training-a.csv', 'training-b.csv', 'training-c.csv',
            'training-d.csv', 'training-e.csv', 'training-f-bit.csv',
            'training-g-color.csv', 'training-h-blackout.csv',
            'training-i-blend.csv', 'training-j-blur.csv',
            'training-k-sap-blur.csv', 'training-l-bs.csv',
            'training-m-sap.csv']
image_path = []
data_frame = pd.DataFrame()

for i in range(np.shape(CSV_LIST)[0]):
    temp_csv_path = os.path.join(MAIN_DIR, CSV_LIST[i])
    temp_data_frame = pd.read_csv(temp_csv_path)
    temp_data_frame = temp_data_frame[['filename', 'digit']]
    data_frame = data_frame.append(temp_data_frame)
data_frame = data_frame.set_index('filename')
print(np.shape(data_frame))


FOLDER_DIR = 'C:/Users/safayet_khan/Desktop/BCR/Numta/numta_dir'
if not os.path.exists(path=FOLDER_DIR):
    os.mkdir(path=FOLDER_DIR)
for i in range(10):
    FOLDER_NAME = FOLDER_DIR + '/' + '{}'.format(i)
    if not os.path.exists(path=FOLDER_NAME):
        os.mkdir(path=FOLDER_NAME)

WRITE_DIR = 'C:/Users/safayet_khan/Desktop/BCR/Numta/numta_flow'
if not os.path.exists(path=WRITE_DIR):
    os.mkdir(path=WRITE_DIR)
for filename, digit in data_frame.iterrows():
    SRC = MAIN_DIR + 'training-{}'.format(filename[0]) + '/' + filename
    DST = WRITE_DIR + '/' + str(digit.digit) + '/' + filename
    shutil.copy(src=SRC, dst=DST)
