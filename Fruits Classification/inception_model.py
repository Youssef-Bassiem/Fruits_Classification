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
from keras.applications import EfficientNetB0
from keras.applications import VGG16
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint
import warnings
from keras.regularizers import l2
from keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import concatenate

x_train = np.load('../../Fruits_DataSet/train_images.npy')
y_train = np.load('../../Fruits_DataSet/train_labels.npy')

IMG_SIZE = 224


def train_model(model, x_train, y_train, epochs=20, validation_split=0.2):
    # Early stopping callback
    early_stopping = EarlyStopping(patience=5, verbose=1, monitor='val_loss', mode='min', restore_best_weights=True)

    # Train the model
    model.fit(x_train, y_train, epochs=epochs, validation_split=validation_split, callbacks=[early_stopping])
    return model


def inception_model():
    input_img = Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    # 1x1 convolution
    conv1x1_1 = Conv2D(8, (1, 1), activation='relu', padding='same')(input_img)

    # 1x1 convolution followed by 3x3 convolution
    conv1x1_2 = Conv2D(8, (1, 1), activation='relu', padding='same')(input_img)
    conv3x3 = Conv2D(4, (3, 3), activation='relu', padding='same')(conv1x1_2)

    # 1x1 convolution followed by 5x5 convolution
    conv1x1_3 = Conv2D(8, (1, 1), activation='relu', padding='same')(input_img)
    conv5x5 = Conv2D(4, (5, 5), activation='relu', padding='same')(conv1x1_3)

    # 3x3 max pooling followed by 1x1 convolution
    maxpool = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(input_img)
    conv1x1_4 = Conv2D(8, (1, 1), activation='relu', padding='same')(maxpool)

    # Concatenate the output of all branches
    inception = concatenate([conv1x1_1, conv3x3, conv5x5, conv1x1_4], axis=-1)

    model = Conv2D(16, (2, 2), activation='relu', padding='same')(inception)

    model = Flatten()(model)

    model = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(model)
    model = Dropout(0.5)(model)
    model = Dense(32, activation='relu')(model)
    model = Dropout(0.2)(model)
    model = Dense(5, activation='softmax')(model)
    model = Model(inputs=input_img, outputs=model)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model


model = inception_model()

model = train_model(model, x_train, y_train, epochs=20, validation_split=0.2)
model.save('../../Fruits_DataSet/inception_model.h5')
