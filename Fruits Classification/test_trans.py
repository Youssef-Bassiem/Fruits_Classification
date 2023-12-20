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
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, Reshape, MultiHeadAttention, LayerNormalization, Dropout
from tensorflow.keras import layers, models

x_train = np.load('../../Fruits_DataSet/train_images.npy')
y_train = np.load('../../Fruits_DataSet/train_labels.npy')

import torch
import torch.nn as nn
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import ToTensor


class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, in_channels=3, embed_dim=768):
        super(PatchEmbedding, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # Input shape: (batch_size, in_channels, image_size, image_size)
        x = self.projection(x)
        # Output shape: (batch_size, embed_dim, num_patches, num_patches)
        x = x.permute(0, 2, 3, 1)
        return x


class ViTBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0):
        super(ViTBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attention_output = self.attention(x, x, x)[0]
        x = x + attention_output
        x = self.norm1(x)
        mlp_output = self.mlp(x)
        x = x + mlp_output
        x = self.norm2(x)
        return x


class VisionTransformer(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, embed_dim=768, num_heads=12, num_layers=12, mlp_ratio=4.0):
        super(VisionTransformer, self).__init__()
        self.patch_embedding = PatchEmbedding(image_size, patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, (image_size // patch_size) ** 2 + 1, embed_dim))
        self.dropout = nn.Dropout(0.1)

        self.blocks = nn.ModuleList([
            ViTBlock(embed_dim, num_heads, mlp_ratio) for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.patch_embedding(x)
        x = x + self.pos_embedding
        x = x.flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        # Extract the class token (cls_token) and use it for classification
        cls_token = x[:, 0, :]
        output = self.fc(cls_token)
        return output


# Example usage:
image_size = 224  # Adjust as needed
patch_size = 16  # Adjust as needed
num_classes = 10  # Adjust based on your classification task

model = VisionTransformer(image_size=image_size, patch_size=patch_size, num_classes=num_classes)

# Display the model summary
print(model)

x_train = np.load('dataSet/train_image.npy')
y_train = np.load('dataSet/train_label.npy')

# Assuming you have x_train and y_train as NumPy arrays
# Convert them to PyTorch tensors
x_train_tensor = torch.from_numpy(x_train).float()
y_train_tensor = torch.from_numpy(y_train).long()

# Create a DataLoader for your dataset
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Initialize the model, loss function, and optimizer
model = VisionTransformer(image_size=224, patch_size=16, num_classes=5)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    average_loss = running_loss / len(train_loader)
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss:.4f}')

# Save the trained model
torch.save(model.state_dict(), 'vision_transformer_model.pth')
print('Model has been saved.')
