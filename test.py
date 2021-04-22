import numpy as np
import pandas as pd
import os
from tensorflow import keras

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ''

# model = keras.models.load_model('./saved_models/ResNet50_01_test/1/ResNet50_01-1-fold-00042.h5')
model = keras.models.load_model('./saved_models/ResNet50_01/1/ResNet50_01-1-fold-00118.h5')
model.summary()

# Get data
train_x = np.load('./data_array_gray/train_x.npy')
train_y = np.load('./data_array_gray/train_y.npy')

train_num = int(len(train_x) * 0.8)
data_index = np.arange(len(train_x))
np.random.seed(0)
np.random.shuffle(data_index)
train_index = data_index[:train_num]
val_index = data_index[train_num:]
print(train_index, len(train_index))
print(val_index, len(val_index))