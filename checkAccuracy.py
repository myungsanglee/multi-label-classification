#-*- coding: utf-8 -*-

import os
import datetime
from tensorflow import keras
import tensorflow as tf
from model import getMultiLabelEfficientNetB7, getMultiLabelResNet50
from generator import DirtyMnistGenerator
import pandas as pd


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
batch_size = 1
lr = 0.001
epoch = 5000

#######################################################
# Set parameters
#######################################################
train_csv_path = '/home/fssv2/dirty_mnist_dataset/dirty_mnist_2nd_answer.csv'
test_csv_path = '/home/fssv2/dirty_mnist_dataset/sample_submission.csv'
train_png_path = '/home/fssv2/dirty_mnist_dataset/dirty_mnist_2nd'
test_png_path = '/home/fssv2/dirty_mnist_dataset/test_dirty_mnist_2nd'


dirty_mnist_answer = pd.read_csv(train_csv_path)
sample_submission = pd.read_csv(test_csv_path)


test_generator = DirtyMnistGenerator(test_png_path, sample_submission, input_shape, num_classes, batch_size, augment=False, shuffle=False)

model = keras.models.load_model('./saved_models/2nd/efficientnetb7_01-00001.h5')
model.summary()

for i in range(test_generator.__len__()):
    batch_x, batch_y = test_generator.__getitem__(i)
    inference = model(batch_x, training=False)
    print(inference)
    print(batch_y)

    break




