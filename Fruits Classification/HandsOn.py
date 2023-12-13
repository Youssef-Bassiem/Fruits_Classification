import os
import cv2
import tensorflow as tf
import numpy as np

def load_test_data():
    test_data = []
    for f in os.listdir('../../Fruits_DataSet/test'):
        img = cv2.imread('../../Fruits_DataSet/test/' + f)
        img = cv2.resize(img, (100, 100))
        # cv2.imshow('', img)
        # cv2.waitKey(0)
        test_data.append(img)
    return test_data



test_data = load_test_data()
model = tf.keras.models.load_model('../../Fruits_DataSet/fruits.h5')
cv2.imshow('', test_data[0])
cv2.waitKey(0)
x = model.predict(test_data[0].reshape(1, 100, 100, 3))
print(x)
