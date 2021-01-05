# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 13:05:42 2020

@author: safayet_khan
"""

import cv2
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

#Load a sample image
SRC = 'C:/Users/safayet_khan/Desktop/BCR/b00002.png'
img_array = cv2.imread(SRC, cv2.IMREAD_COLOR)
img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

pixel_colors = img_array.reshape(np.shape(img_array)[0] *
                                 np.shape(img_array)[1], 3)
norm = colors.Normalize(vmin=-1, vmax=1.)
norm.autoscale(pixel_colors)
pixel_colors = norm(pixel_colors).tolist()

hsv_img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
h, s, v = cv2.split(hsv_img_array)

plt1 = plt.figure(num=1)
axis = plt1.add_subplot(1, 1, 1, projection="3d")
axis.scatter(h.flatten(), s.flatten(), v.flatten(),
              facecolors=pixel_colors, marker=".")
axis.set_xlabel("Hue")
axis.set_ylabel("Saturation")
axis.set_zlabel("Value")
plt.show()

DARK_BLACK = (0, 0, 0)
LIGHT_BLACK = (255, 255, 160)
CHANNEL_VALUE = [100, 0, 0]
mask = cv2.inRange(hsv_img_array, DARK_BLACK, LIGHT_BLACK)
img_array[mask==255] = CHANNEL_VALUE

plt2 = plt.figure(num=2)
plt.subplot(1, 2, 1)
plt.imshow(mask, cmap="gray")
plt.subplot(1, 2, 2)
plt.imshow(img_array)
plt.show()
