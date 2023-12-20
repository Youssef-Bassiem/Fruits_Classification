from keras.callbacks import EarlyStopping
import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.models import Model
from keras.layers import Input
import numpy as np
import cv2
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.regularizers import l2
from keras.layers import GlobalAveragePooling2D

x_train = np.load('../../Fruits_DataSet/train_images.npy')
y_train = np.load('../../Fruits_DataSet/train_labels.npy')

print(x_train[0].shape)
print(y_train.shape)
IMG_SIZE = 224
num_classes = 5


def create_vgg16_model(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=num_classes):
    model = Sequential()

    # Block 1
    model.add(Conv2D(8, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Block 2
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Block 3
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Block 4
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Block 5
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Flatten the output for Dense layers
    model.add(Flatten())

    # Fully connected layers
    model.add(Dense(512, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))

    # Compile the model
    model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def train_vgg16_model(model, x_train, y_train, epochs=25, validation_split=0.3):
    # Assuming that your labels are in integer format, convert them to one-hot encoding
    # y_train = to_categorical(y_train)

    # Early stopping callback
    early_stopping = EarlyStopping(patience=5, verbose=1, monitor='val_loss', mode='min', restore_best_weights=True)

    # Train the model
    model.fit(x_train, y_train, epochs=epochs, validation_split=validation_split, callbacks=[early_stopping])

    return model


vgg16_model = create_vgg16_model()
trained_model = train_vgg16_model(vgg16_model, x_train, y_train)
trained_model.save('../../Fruits_DataSet/vgg16_model.h5')
