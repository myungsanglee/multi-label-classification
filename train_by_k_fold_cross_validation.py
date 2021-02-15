# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow import keras
import datetime
import os
from sklearn.model_selection import KFold
from glob import glob
from model import getMultiLabelEfficientNetB7
from generator import DirtyMnistGeneratorV2
import pandas as pd
import numpy as np
from tqdm import tqdm

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
batch_size = 56
test_batch_size = 20
lr = 0.001
epoch = 50000

#######################################################
# Set parameters
#######################################################
# Get Data array
train_x = np.load('./train_x.npy')
train_y = np.load('./train_y.npy')
test_x = np.load('./test_x.npy')
test_y = np.load('./test_y.npy')

# Get Test generator
test_generator = DirtyMnistGeneratorV2(test_x, test_y, test_batch_size, augment=False, shuffle=False)

# model list parameter
model_list = [getMultiLabelEfficientNetB7]

# Set StratifiedKFold
kfold = KFold(n_splits=5, shuffle=True, random_state=0)

# parameters
model_name_list = ['EfficientNetB7']

# txt file
# with open('./result.txt', 'w') as file:
#     file.write('Result of Training\n\n')

# 모델에 대해서 각각 K-fold cross validation 을 진행한다.
for model_index, model_function in enumerate(model_list):
    # parameters
    model_name = model_name_list[model_index]

    # best model list
    best_models = []

    # k-fold validation
    for fold_index, (train_index, val_index) in enumerate(kfold.split(train_x, train_y)):
        # Set train dataset
        train_data_x = train_x[train_index]
        train_data_y = train_y[train_index]
        train_generator = DirtyMnistGeneratorV2(train_data_x, train_data_y, batch_size, augment=True, shuffle=True)

        # Set val dataset
        val_data_x = train_x[val_index]
        val_data_y = train_y[val_index]
        val_generator = DirtyMnistGeneratorV2(val_data_x, val_data_y, batch_size, augment=False, shuffle=False)

        # Set callbacks
        log_dir = os.path.join(
            "logs", "fit",
            datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '-' + model_name + '-' + str(fold_index + 1) + '-fold',
        )

        save_model_path = './saved_models/' + model_name + '/' + str(fold_index + 1)
        m_name = model_name + '-' + str(fold_index + 1) + '-fold'

        callbacks = [keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                       factor=0.75,
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

        with strategy.scope():
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
                  workers=1,
                  use_multiprocessing=False
                  )

        # Append best model path
        best_model = glob(save_model_path + '/*')
        best_model = sorted(best_model)
        best_models.append(best_model[-1])

        # Save result
        # current_model = str(best_model[-1]) + ' Model Result\n'
        # current_model_result = 'Val Loss: {0:.4f}, Val Accuracy: {1:.2f}%\n'.format(eval_result[0], eval_result[1] * 100)
        # with open('./result.txt', 'a') as file:
        #     file.write(current_model)
        #     file.write(current_model_result)
        #
        # print(current_model)
        # print(current_model_result)

        # Clear Session
        keras.backend.clear_session()

    # 5개의 fold마다 가장 좋은 모델을 이용하여 예측
    print(best_models)
    predictions_list = []
    for model_path in best_models:
        prediction_array = np.zeros(shape=(test_y.shape[0], test_y.shape[1]))
        model = keras.models.load_model(model_path)

        for i in tqdm(range(test_generator.__len__())):
            batch_x, batch_y = test_generator.__getitem__(i)
            inference = model(batch_x, training=False)
            # print(inference)
            inference = inference.numpy()
            inference = (inference > 0.5)
            # print(inference)
            inference = inference.astype(int)
            # print(inference)

            batch_index = test_batch_size * i
            prediction_array[batch_index:batch_index + test_batch_size, :] = inference

        prediction_array = np.expand_dims(prediction_array, axis=-1)
        predictions_list.append(prediction_array)

        # Clear Session
        keras.backend.clear_session()

    # axis = 2를 기준으로 평균
    predictions_array = np.concatenate(predictions_list, axis=2)
    predictions_mean = predictions_array.mean(axis=2)

    # 평균 값이 0.5보다 클 경우 1 작으면 0
    predictions_mean = (predictions_mean > 0.5) * 1

    # 제출파일 생성
    sample_submission = pd.read_csv('/home/fssv2/dirty_mnist_dataset/sample_submission.csv')
    sample_submission.iloc[:, 1:] = predictions_mean
    csv_file_path = './' + model_name + '.csv'
    sample_submission.to_csv(csv_file_path, index=False)












