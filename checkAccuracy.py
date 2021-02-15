#-*- coding: utf-8 -*-

import os
import datetime
from tensorflow import keras
import tensorflow as tf
from model import getMultiLabelEfficientNetB7, getMultiLabelResNet50_01
from generator import DirtyMnistGeneratorV1
import pandas as pd
from tqdm import tqdm
import numpy as np


#######################################################
# Set GPU
#######################################################
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
# GPU를 아예 못 보게 하려면: os.environ["CUDA_VISIBLE_DEVICES"]=''
# GPU 0만 보게 하려면: os.environ["CUDA_VISIBLE_DEVICES"]='0'
# GPU 1만 보게 하려면: os.environ["CUDA_VISIBLE_DEVICES"]='0'
# GPU 0과 1을 보게 하려면: os.environ["CUDA_VISIBLE_DEVICES"]='0,1'

#######################################################
# Set Memory
#######################################################
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
# if gpus:
#   try:
#     tf.config.experimental.set_virtual_device_configuration(
#         gpus[0],
#         [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
#   except RuntimeError as e:
#     print(e)

#######################################################
# Set hyper parameters
#######################################################
input_shape = (256, 256, 1)
num_classes = 26
batch_size = 20
lr = 0.001
epoch = 5000

#######################################################
# Set parameters
#######################################################
# train_csv_path = '/home/fssv2/dirty_mnist_dataset/dirty_mnist_2nd_answer.csv'
# test_csv_path = '/home/fssv2/dirty_mnist_dataset/sample_submission.csv'
# train_png_path = '/home/fssv2/dirty_mnist_dataset/dirty_mnist_2nd'
# test_png_path = '/home/fssv2/dirty_mnist_dataset/test_dirty_mnist_2nd'
#
#
# dirty_mnist_answer = pd.read_csv(train_csv_path)
# sample_submission = pd.read_csv(test_csv_path)
# test_submission = sample_submission.copy()
#
#
# test_generator = DirtyMnistGeneratorV1(test_png_path, sample_submission, input_shape, num_classes, batch_size, augment=False, shuffle=False)
#
model = keras.models.load_model('./saved_models/EfficientNetB7/2/EfficientNetB7-2-fold-00053.h5')
model.summary()


from generator import DirtyMnistGeneratorV2
test_x = np.load('./test_x.npy')
test_y = np.load('./test_y.npy')
test_generator = DirtyMnistGeneratorV2(test_x, test_y, batch_size, augment=False, shuffle=False)

predictions_list = []
prediction_array = np.zeros(shape=(test_y.shape[0], test_y.shape[1]))
print(prediction_array.shape)

for i in tqdm(range(test_generator.__len__())):
    batch_x, batch_y = test_generator.__getitem__(i)
    inference = model(batch_x, training=False)
    # print(inference)
    inference = inference.numpy()
    inference = (inference > 0.5)
    # print(inference)
    inference = inference.astype(int)
    # print(inference)

    batch_index = batch_size * i
    prediction_array[batch_index:batch_index + batch_size, :] = inference

prediction_array = np.expand_dims(prediction_array, axis=-1)
predictions_list.append(prediction_array)



#     for j in range(batch_size):
#         test_submission.iloc[j + (i * batch_size), 1:] = inference[j]
#
# test_submission.to_csv('test.csv', index=False)

# for i in tqdm(range(test_generator.__len__())):
#     batch_x, batch_y = test_generator.__getitem__(i)
#     inference = model(batch_x, training=False)
#     # print(inference)
#     inference = inference.numpy()
#     inference = (inference > 0.5)
#     # print(inference)
#     inference = inference.astype(int)
#     # print(inference)
#     for j in range(batch_size):
#         test_submission.iloc[j + (i * batch_size), 1:] = inference[j]
#
# test_submission.to_csv('test.csv', index=False)



