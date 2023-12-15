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

# Filter specific warning
warnings.filterwarnings("ignore", category=DeprecationWarning, module="tensorflow")

x_train = np.load('../../Fruits_DataSet/train_images.npy')
y_train = np.load('../../Fruits_DataSet/train_labels.npy')

print(x_train[0].shape)
print(y_train.shape)
IMG_SIZE = 170
num_classes = 5


# Learning Rate Scheduler
def scheduler(epoch, lr):
    if epoch < 5:
        return lr
    else:
        return lr * tf.math.exp(-0.1)


def train_model(model, x_train, y_train, epochs=10, validation_split=0.2):
    lr_scheduler = LearningRateScheduler(scheduler)

    # Early stopping callback
    early_stopping = EarlyStopping(patience=5, verbose=1, monitor='val_loss', mode='min', restore_best_weights=True)

    # Model Checkpoint
    checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True)

    # Train the model
    model.fit(x_train, y_train, epochs=epochs, validation_split=validation_split,
              callbacks=[early_stopping, early_stopping, checkpoint])

    return model


def create_transformer_cnn_model(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=num_classes):
    # Create a base transformer-based model
    transformer_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)

    # Freeze the layers of the transformer model
    for layer in transformer_model.layers:
        layer.trainable = False

    # Create a sequential model with the transformer base
    model = Sequential()
    model.add(transformer_model)

    # Additional CNN layers
    model.add(Conv2D(512, (3, 3), activation='relu'))
    # model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    # model.add(Conv2D(64, (3, 3), activation='relu'))
    # model.add(MaxPooling2D((2, 2)))
    # model.add(Conv2D(32, (3, 3), activation='relu'))
    # model.add(MaxPooling2D((2, 2)))

    # Flatten the output for Dense layers
    model.add(Flatten())

    # Fully connected layers (you can customize this part)
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    return model


transformer_cnn_model = create_transformer_cnn_model()
trained_model_transformer = train_model(transformer_cnn_model, x_train, y_train)
trained_model_transformer.save('transformer_cnn_model.h5')

