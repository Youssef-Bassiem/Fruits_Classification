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
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
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

x_train = np.load('../../Fruits_DataSet/train_images.npy')
y_train = np.load('../../Fruits_DataSet/train_labels.npy')


def unfreeze_model(model):
    for layer in model.layers:
        layer.trainable = True


def train_model(model, x_train, y_train, epochs=20, validation_split=0.3):
    # Early stopping callback
    early_stopping = EarlyStopping(patience=5, verbose=1, monitor='val_loss', mode='min', restore_best_weights=True)

    # Train the model
    model.fit(x_train, y_train, epochs=epochs, validation_split=validation_split)
    return model


# Load the previously trained model
model = load_model('../../Fruits_DataSet/vgg16_model.h5')

# Unfreeze all layers
unfreeze_model(model)

# Compile the model (necessary after modifying trainable property)
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model with unfrozen layers
epochs_fine_tune = 20  # Adjust the number of fine-tuning epochs as needed
fine_tuned_model = train_model(model, x_train, y_train, epochs=epochs_fine_tune)

# Save the fine-tuned model
fine_tuned_model.save('../../Fruits_DataSet/improved_vgg16_model.h5')

