from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Function to create Vision Transformer model
'''
image_size:
  The size of the input images.
num_classes:
  The number of output classes.
patch_size:
  The size of the patches to be extracted from the input images.
num_patches:
  The number of patches in the image.
projection_dim:
  The dimensionality of the projected patches.
num_heads:
  The number of attention heads in the Multi-Head Attention layer.
transformer_units:
   A list specifying the number of units in each transformer block.
num_blocks:
  The number of transformer blocks in the model.
mlp_head_units:
  A list specifying the number of units in the MLP head.
dropout_rate:
  The dropout rate used in the model. Default is set to 0.1.

'''


# Function to create MLP
def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        # The activation function used is the GELU (Gaussian Error Linear Unit) activation
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


def vision_transformer(patch_size, num_patches, projection_dim, num_heads, transformer_units,
                       num_blocks, mlp_head_units, dropout_rate=0.1):
    inputs = tf.keras.Input(shape=(224, 224, 3))

    # Patch extraction
    patches = tf.image.extract_patches(
        images=inputs,
        sizes=[1, patch_size, patch_size, 1],
        strides=[1, patch_size, patch_size, 1],
        rates=[1, 1, 1, 1],
        padding='VALID',
    )

    batch_size = tf.shape(patches)[0]
    patch_dims = patches.shape[-1]
    patches = tf.reshape(patches, [batch_size, -1, patch_dims])
    # patches = Flatten()(patches)

    # Patch embedding and positional encoding
    positions = tf.range(start=0, limit=num_patches, delta=1)
    encoded_patches = layers.Dense(units=projection_dim)(patches) + layers.Embedding(input_dim=num_patches,
                                                                                     output_dim=projection_dim)(
        positions)

    # Transformer blocks
    for _ in range(num_blocks):
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim, dropout=dropout_rate)(
            x1, x1)
        x2 = layers.Add()([attention_output, encoded_patches])
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=dropout_rate)
        encoded_patches = layers.Add()([x3, x2])

    # # Global average pooling
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)

    # # MLP head
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=dropout_rate)

    # Output layer
    probs = layers.Dense(5, activation='softmax')(features)

    # Create the model
    model = tf.keras.Model(inputs=inputs, outputs=probs)

    return model


x_train = np.load('../../Fruits_DataSet/train_images.npy')
y_train = np.load('../../Fruits_DataSet/train_labels.npy')

learning_rate = 0.001
num_epochs = 10

patch_size = 16
num_patches = (224 // patch_size) ** 2
projection_dim = 64
num_heads = 4
transformer_units = [projection_dim * 2, projection_dim]  # Size of the transformer layers
num_blocks = 8
mlp_head_units = [256, 5]
dropout_rate = 0.1

vit_model = vision_transformer(patch_size, num_patches, projection_dim, num_heads,
                               transformer_units, num_blocks, mlp_head_units, dropout_rate)

# Compile the model
vit_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss='categorical_crossentropy',
                  metrics=['accuracy'])
vit_model.summary()
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
vit_model.fit(x_train, y_train, epochs=num_epochs, validation_split=0.2, callbacks=[early_stopping])
vit_model.save('../../Fruits_DataSet/vision_transformer_model.h5')

