#-*- coding: utf-8 -*-

import os
import datetime
from tensorflow import keras
import tensorflow as tf
from model import getMultiLabelEfficientNetB7, getMultiLabelResNet50_01
from generator import DirtyMnistGeneratorV1, DirtyMnistGeneratorV2
import pandas as pd
import numpy as np


#######################################################
# Set GPU
#######################################################
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
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

strategy = tf.distribute.MirroredStrategy()

#######################################################
# Set hyper parameters
#######################################################
input_shape = (256, 256, 1)
num_classes = 26
batch_size = 64
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

m_name = 'ResNet50_01-01'

log_dir = os.path.join(
    "logs", "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '-' + m_name,
)

#######################################################
# Get generator
#######################################################
train_x = np.load('./train_x.npy')
train_y = np.load('./train_y.npy')
# train_generator = PassengerFaceGenerator(train_x, train_y, input_shape, num_classes, batch_size, augment=True, shuffle=True)
# val_generator = PassengerFaceGenerator(val_x, val_y, input_shape, num_classes, batch_size, augment=False, shuffle=False)
# train_generator = DirtyMnistGeneratorV1(train_png_path, dirty_mnist_answer, input_shape, num_classes, batch_size, augment=False, shuffle=False)
train_generator = DirtyMnistGeneratorV2(train_x, train_y, batch_size, augment=False, shuffle=False)
# val_generator = DirtyMnistGenerator(train_png_path, dirty_mnist_answer, input_shape, num_classes, batch_size, augment=False, shuffle=False)


with strategy.scope():

    #######################################################
    # Get Model
    #######################################################
    # model = getMultiLabelEfficientNetB7(input_shape, num_classes)
    model = getMultiLabelResNet50_01(input_shape, num_classes)
    model.summary()

    #######################################################
    # Compile Model
    #######################################################
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                  loss=keras.losses.binary_crossentropy,
                  metrics=['accuracy'])

#######################################################
# Set callbacks
#######################################################
callbacks = [keras.callbacks.ReduceLROnPlateau(monitor='loss',
                                               factor=0.75,
                                               patience=5,
                                               verbose=1,
                                               mode='min',
                                               min_lr=0.0001),
             keras.callbacks.ModelCheckpoint(filepath='./saved_models/' + m_name + '-{epoch:05d}.h5',
                                             monitor='loss',
                                             verbose=1,
                                             save_best_only=True,
                                             mode='min',
                                             save_freq='epoch'),
             keras.callbacks.TensorBoard(log_dir=log_dir,
                                         profile_batch=0),
             keras.callbacks.EarlyStopping(monitor='loss',
                                           patience=25,
                                           verbose=1,
                                           mode='min')
             ]

#######################################################
# Train Model
#######################################################
model.fit(x=train_generator,
          epochs=epoch,
          verbose=1,
          # callbacks=callbacks,
          # validation_data=val_generator,
          max_queue_size=10,
          workers=1,
          use_multiprocessing=False
          )
