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
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint

from keras.regularizers import l2
from keras.layers import GlobalAveragePooling2D
from keras.layers import MultiHeadAttention, LayerNormalization
from keras.layers import Concatenate
from keras.layers import Attention

x_train = np.load('../../Fruits_DataSet/train_images.npy')
y_train = np.load('../../Fruits_DataSet/train_labels.npy')

IMG_SIZE = 224
num_classes = 5


def train_model(model, x_train, y_train, epochs=5, validation_split=0.3):
    # Early stopping callback
    early_stopping = EarlyStopping(patience=5, verbose=1, monitor='val_loss', mode='min', restore_best_weights=True)

    # Train the model
    model.fit(x_train, y_train, epochs=epochs, validation_split=validation_split, callbacks=[early_stopping])
    return model


def create_transformer_model(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=num_classes):
    # Create a base transformer-based model
    transformer_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)

    # Freeze the layers of the transformer model
    for layer in transformer_model.layers:
        layer.trainable = False

    transformer_model.layers[-1].trainable = True
    transformer_model.layers[-2].trainable = True
    transformer_model.layers[-3].trainable = True
    transformer_model.layers[-4].trainable = True

    # Create a sequential model with the transformer base
    model = Sequential()
    # model.add(transformer_model)

    # transformer_model.summary()

    model.add(Conv2D(8, (3, 3), activation='relu', padding='same', input_shape=input_shape))

    attention_output = MultiHeadAttention(num_heads=5, key_dim=512)(model.output)

    # LayerNormalization
    attention_output = LayerNormalization(epsilon=1e-6)(attention_output)

    # Connect the attention output to the transformer model
    model.add(attention_output)

    # Global Average Pooling
    model.add(GlobalAveragePooling2D())

    # Fully connected layers (you can customize this part)
    model.add(Dense(512, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


transformer_model = create_transformer_model()
trained_transformer_model = train_model(transformer_model, x_train, y_train)
trained_transformer_model.save('../../Fruits_DataSet/transformer_model_true.h5')
