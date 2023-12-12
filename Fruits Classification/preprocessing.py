import os
import cv2
import pandas as pd
import numpy as np
from random import shuffle


def resize_img(img):
    img = cv2.resize(img, (200, 200))
    return img


def create_label(image_class):
    if image_class == '1':
        return np.array([1, 0, 0, 0, 0])
    elif image_class == '2':
        return np.array([0, 1, 0, 0, 0])
    elif image_class == '3':
        return np.array([0, 0, 1, 0, 0])
    elif image_class == '4':
        return np.array([0, 0, 0, 1, 0])
    elif image_class == '5':
        return np.array([0, 0, 0, 0, 1])


data = []
path = 'dataset/train'
for i in os.listdir(path):
    for j in os.listdir(path + '/' + i):
        if i in ['1', '2', '3', '4', '5']:
            img = cv2.imread(path + '/' + i + '/' + j, cv2.IMREAD_COLOR)
            img = resize_img(img)
            data.append((img, create_label(i)))
            data.append((cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE), create_label(i)))
            data.append((cv2.flip(img, -1), create_label(i)))
            data.append((cv2.flip(img, 0), create_label(i)))
            data.append((cv2.flip(img, 1), create_label(i)))

# Convert the list of tuples into separate arrays
images, labels = zip(*data)
images = np.array(images)
labels = np.array(labels)

# Shuffle the data
indices = np.arange(len(images))
np.random.shuffle(indices)
images = images[indices]
labels = labels[indices]

# Save the arrays
np.save('dataset/train_images.npy', images)
np.save('dataset/train_labels.npy', labels)

print('done')
