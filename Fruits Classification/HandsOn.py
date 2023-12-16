import os
import cv2
import pandas as pd
import tensorflow as tf
import numpy as np
import csv

output = pd.DataFrame(columns=['image_id', 'label'])


def load_test_data():
    test_data = []
    for f in os.listdir('../../Fruits_DataSet/test'):
        img = cv2.imread('../../Fruits_DataSet/test/' + f)
        img = cv2.resize(img, (170, 170))
        test_data.append((img, f.split('.')[0]))
        # print(test_data[-1][1])
    return test_data


test_data = load_test_data()
model = tf.keras.models.load_model('../../Fruits_DataSet/transformer_cnn_model.h5')
labels = ['apple', 'moz', '3nab', 'mango', 'farawla']
for i in range(len(test_data)):
    x = model.predict(test_data[i][0].reshape(1, 170, 170, 3))
    new_row = {'image_id': test_data[i][1], 'label': np.argmax(x) + 1}
    output.loc[len(output)] = new_row
    # print(labels[np.argmax(x)])
    # cv2.imshow('', test_data[i])
    # cv2.waitKey(0)
output.to_csv('transformer_model_test.csv', index=False)

# ***********************test**************************************

y_test = '../../Fruits_DataSet/y_test.csv'
x_test = '../../Fruits_DataSet/transformer_model_test.csv'

with open(y_test, 'r') as file1:
    csv_reader1 = csv.reader(file1)

    with open(x_test, 'r') as file2:
        csv_reader2 = csv.reader(file2)
        next(csv_reader1)
        next(csv_reader2)
        mismatch_count = 0

        for row1, row2 in zip(csv_reader1, csv_reader2):
            if row1 != row2:
                print(f"Mismatch at row {csv_reader1.line_num}: {row1} != {row2}")
                mismatch_count += 1

print(f"Total number of mismatched rows: {mismatch_count}")

# ***********************test**************************************
