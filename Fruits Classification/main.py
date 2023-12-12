import os
import cv2
import pandas as pd
import numpy as np
from random import shuffle

train_data = np.load('dataset/train_images.npy')
train_labels = np.load('dataset/train_labels.npy')
for i in range(10):
    cv2.imshow('fruit', train_data[i])
    print(train_labels[i])
    cv2.waitKey(0)
