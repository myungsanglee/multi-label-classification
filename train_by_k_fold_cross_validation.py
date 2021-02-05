# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow import keras
import datetime
import os
from sklearn.model_selection import KFold
from glob import glob
from model import getMultiLabelResNet50, getMultiLabelEfficientNetB7
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
batch_size = 16
lr = 0.001
epoch = 5000

#######################################################
# Set parameters
#######################################################
# path parameters
train_csv_path = '/home/fssv2/dirty_mnist_dataset/dirty_mnist_2nd_answer.csv'
# test_csv_path = '/home/fssv2/dirty_mnist_dataset/sample_submission.csv'
train_png_path = '/home/fssv2/dirty_mnist_dataset/dirty_mnist_2nd'

# Get pandas dataframe
dirty_mnist_answer = pd.read_csv(train_csv_path)
# sample_submission = pd.read_csv(test_csv_path)

# model list parameter
model_list = [getMultiLabelEfficientNetB7, getMultiLabelResNet50]

# Set StratifiedKFold
kfold = KFold(n_splits=10, shuffle=True, random_state=0)

# parameters
model_name_list = ['EfficientNetB7', 'ResNet50']

# txt file
with open('./result.txt', 'w') as file:
    file.write('Result of Training\n\n')

# 10개의 모델에 대해서 각각 K-fold cross validation 을 진행한다.
# validation Accuracy 평균을 기준으로 제일 좋은 모델을 선택한다.
for model_index, model_function in enumerate(model_list):
    # parameters
    model_name = model_name_list[model_index]

    # k-fold validation
    for fold_index, (train_index, val_index) in enumerate(kfold.split(dirty_mnist_answer)):
        # Set train dataset
        train_answer = dirty_mnist_answer.iloc[train_index]
        train_generator = DirtyMnistGenerator(train_png_path, train_answer, input_shape, num_classes, batch_size,
                                              augment=True, shuffle=True)

        # Set val dataset
        val_answer = dirty_mnist_answer.iloc[val_index]
        val_generator = DirtyMnistGenerator(train_png_path, val_answer, input_shape, num_classes, batch_size,
                                            augment=False, shuffle=False)

        # Set callbacks
        log_dir = os.path.join(
            "logs", "fit",
            datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '-' + model_name + '-' + str(fold_index + 1) + '-fold',
        )

        save_model_path = './saved_models/' + model_name + '/' + str(fold_index + 1)
        m_name = model_name + '-' + str(fold_index + 1) +'-fold'

        callbacks = [keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                       factor=0.5,
                                                       patience=5,
                                                       verbose=1,
                                                       mode='min',
                                                       min_lr=0.0001),
                     keras.callbacks.ModelCheckpoint(
                         filepath=save_model_path + '/' + m_name + '-{epoch:05d}.h5',
                         monitor='val_loss',
                         verbose=1,
                         save_best_only=True,
                         mode='min',
                         save_freq='epoch'),
                     keras.callbacks.TensorBoard(log_dir=log_dir,
                                                 profile_batch=0),
                     keras.callbacks.EarlyStopping(monitor='val_loss',
                                                   patience=25,
                                                   verbose=1,
                                                   mode='min')
                     ]

        # Get model
        model = model_function(input_shape, num_classes)
        model.summary()

        # Compile Model
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                      loss=keras.losses.binary_crossentropy,
                      metrics=['accuracy'])

        # Train Model
        model.fit(x=train_generator,
                  epochs=epoch,
                  verbose=1,
                  callbacks=callbacks,
                  validation_data=val_generator,
                  max_queue_size=10,
                  workers=8,
                  use_multiprocessing=False
                  )

        # Load Best model
        best_model = glob(save_model_path + '/*')
        best_model = sorted(best_model)
        model.load_weights(best_model[-1])

        # Evaluate Model
        eval_result = model.evaluate(x=val_generator,
                                     verbose=1,
                                     max_queue_size=10,
                                     workers=8,
                                     use_multiprocessing=False
                                     )

        # Save result
        current_model = str(best_model[-1]) + ' Model Result\n'
        current_model_result = 'Val Loss: {0:.4f}, Val Accuracy: {1:.2f}%\n'.format(eval_result[0], eval_result[1] * 100)
        with open('./result.txt', 'a') as file:
            file.write(current_model)
            file.write(current_model_result)

        print(current_model)
        print(current_model_result)

        # Clear Session
        keras.backend.clear_session()
















