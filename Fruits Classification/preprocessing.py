import os
import cv2
import pandas as pd
import numpy as np
from random import shuffle
import random


def resize_img(img):
    img = cv2.resize(img, (224, 224))
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


def random_zoom_and_crop(img):
    zoom_factor = random.uniform(0.8, 1.2)

    new_width = int(img.shape[1] * zoom_factor)
    new_height = int(img.shape[0] * zoom_factor)

    start_x = random.randint(0, max(0, new_width - img.shape[1]))
    start_y = random.randint(0, max(0, new_height - img.shape[0]))

    zoomed_img = img[start_y:(start_y + img.shape[0]), start_x:(start_x + img.shape[1])]
    zoomed_img = cv2.resize(zoomed_img, (img.shape[1], img.shape[0]))

    return zoomed_img


def random_brightness_contrast(img):
    alpha = random.uniform(0.8, 1.2)
    beta = random.uniform(-20, 20)

    adjusted_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return adjusted_img


data = []
path = '../../Fruits_DataSet/train'

augmentation_probability_flip = 0.2
augmentation_probability_zoom = 0.3
for i in os.listdir(path):
    for j in os.listdir(path + '/' + i):
        if i in ['1', '2', '3', '4', '5']:
            img = cv2.imread(path + '/' + i + '/' + j, cv2.IMREAD_COLOR)
            img = resize_img(img)
            data.append((img, create_label(i)))

            data.append((cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE), create_label(i)))
            if random.random() < augmentation_probability_flip:
                data.append((cv2.flip(img, -1), create_label(i)))
                data.append((cv2.flip(img, 0), create_label(i)))
                data.append((cv2.flip(img, 1), create_label(i)))

            if random.random() < augmentation_probability_zoom:
                zoomed_img = random_zoom_and_crop(img)
                data.append((zoomed_img, create_label(i)))

            bright_contrast_img = random_brightness_contrast(img)
            data.append((bright_contrast_img, create_label(i)))

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
np.save('../../Fruits_DataSet/train_images.npy', images)
np.save('../../Fruits_DataSet/train_labels.npy', labels)

print('done')
