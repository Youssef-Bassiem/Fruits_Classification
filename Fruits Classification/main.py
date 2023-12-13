from keras.callbacks import EarlyStopping
import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.models import Model
from keras.layers import Input
import numpy as np
import cv2

x_train = np.load('../../dataset/train_images.npy')
y_train = np.load('../../dataset/train_labels.npy')

print(x_train[0].shape)
print(y_train.shape)
IMG_SIZE = 100

# def cnn_model():
#     model = Sequential()
#     model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)))
#     model.add(MaxPooling2D((2, 2)))
#     model.add(Conv2D(64, (3, 3), activation='relu'))
#     model.add(MaxPooling2D((2, 2)))
#     model.add(Conv2D(128, (3, 3), activation='relu'))
#     model.add(MaxPooling2D((2, 2)))
#     model.add(Conv2D(128, (3, 3), activation='relu'))
#     model.add(MaxPooling2D((2, 2)))
#     model.add(BatchNormalization())
#     model.add(Flatten())
#     model.add(Dropout(0.5))
#     model.add(Dense(512, activation='relu'))
#     model.add(BatchNormalization())
#     model.add(Dense(6, activation='softmax'))
#
#     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#     model.summary()
#
#     if os.path.exists('sports.h5'):
#         model = tf.keras.models.load_model('sports.h5')
#     else:
#         early_stopping = EarlyStopping(patience=5, verbose=1, monitor='val_loss', mode='min', restore_best_weights=True)
#         history = model.fit(x_train, y_train, epochs=50, validation_split=0.2, callbacks=[early_stopping])
#         model.save('sports.h5')
#     return model


def inception_model():
    input_img = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    conv1 = Conv2D(10, (1, 1), activation='relu', padding='same')(input_img)
    conv2 = Conv2D(10, (1, 1), activation='relu', padding='same')(input_img)
    # In inception v1 should be 1x1 filter before 3x3 to reduce the number of math operations
    conv2 = Conv2D(10, (3, 3), activation='relu', padding='same')(conv2)

    conv3 = Conv2D(10, (1, 1), activation='relu', padding='same')(input_img)
    # In inception v1 should be 1x1 before 5x5
    # In inception v2 5x5 filter should be 3x3 twice
    conv3 = Conv2D(10, (3, 3), activation='relu', padding='same')(conv3)
    conv3 = Conv2D(10, (3, 3), activation='relu', padding='same')(conv3)

    max_pool = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(input_img)
    max_pool = Conv2D(10, (1, 1), activation='relu', padding='same')(max_pool)

    concatenation = tf.keras.layers.concatenate([conv1,conv2, conv3, max_pool], axis=-1)
    conv4 = Conv2D(10, (1, 1), activation='relu', padding='same')(concatenation)
    conv4 = Conv2D(10, (3, 3), activation='relu', padding='same')(conv4)

    conv5 = Conv2D(10, (1, 1), activation='relu', padding='same')(concatenation)
    conv5 = Conv2D(10, (3, 3), activation='relu', padding='same')(conv5)
    conv5 = Conv2D(10, (3, 3), activation='relu', padding='same')(conv5)

    max_pool2 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(concatenation)
    max_pool2 = Conv2D(10, (1, 1), activation='relu', padding='same')(max_pool2)
    concatenation1 = tf.keras.layers.concatenate([conv4, conv5, max_pool2], axis=-1)

    drop = Dropout(0.5)(concatenation1)
    batch = BatchNormalization()(drop)
    flat = Flatten()(batch)
    dense = Dense(128, activation='relu')(flat)
    output = Dense(5, activation='softmax')(dense)
    model = Model(inputs=input_img, outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    # if os.path.exists('sports.h5'):
    #     model = tf.keras.models.load_model('sports.h5')
    # else:
    early_stopping = EarlyStopping(patience=5, verbose=1, monitor='val_loss', mode='min', restore_best_weights=True)
    model.fit(x_train, y_train, epochs=5, validation_split=0.2, callbacks=[early_stopping])
    model.save('../../fruits.h5')
    # return model


# model = cnn_model()
model = inception_model()

# with open('submission_file.csv', 'w') as f:
#     f.write('image_name,label\n')
#     for i in os.listdir('Test'):
#         img = cv2.imread('Test/' + i, cv2.IMREAD_COLOR)
#         test_img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
#         test_img = test_img.reshape(IMG_SIZE, IMG_SIZE, 3)
#         test_img = test_img / 255.0
#
#         prediction = model.predict(np.array([test_img]))
#
#         if np.argmax(prediction) == 0:
#             f.write(i + ',' + '0')
#         elif np.argmax(prediction) == 1:
#             f.write(i + ',' + '1')
#         elif np.argmax(prediction) == 2:
#             f.write(i + ',' + '2')
#         elif np.argmax(prediction) == 3:
#             f.write(i + ',' + '3')
#         elif np.argmax(prediction) == 4:
#             f.write(i + ',' + '4')
#         elif np.argmax(prediction) == 5:
#             f.write(i + ',' + '5')
#         f.write('\n')