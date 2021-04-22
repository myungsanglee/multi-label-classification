#######################################################
# Import library
#######################################################
import tensorflow as tf
from tensorflow import keras
import datetime
import os
from glob import glob
from model import getMultiLabelResNet50_11
from generator import DirtyMnistGeneratorV2
import numpy as np
from send_mail import sendMail
from checkAccuracy import getResultOfTest
import traceback
import logging

# logging parameter
logging.basicConfig(level=logging.ERROR)

#######################################################
# Set GPU
#######################################################
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

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


def main():
    try:
        #######################################################
        # Set hyper parameters
        #######################################################
        input_shape = (256, 256, 1)
        num_classes = 26
        batch_size = 128
        lr = 0.001
        epoch = 200
        strategy = tf.distribute.MirroredStrategy()

        #######################################################
        # Set parameters
        #######################################################
        # parameters
        model_name = 'ResNet50_11'
        save_model_path = './saved_models/' + model_name
        log_dir = os.path.join(
            "logs", "fit",
            datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '-' + model_name,
        )
        best_models = []

        #######################################################
        # Get Data
        #######################################################
        # Get data
        train_x = np.load('./data_array_gray/train_x.npy')
        train_y = np.load('./data_array_gray/train_y.npy')

        train_num = int(len(train_x) * 0.8)
        data_index = np.arange(len(train_x))
        np.random.seed(0)
        np.random.shuffle(data_index)
        train_index = data_index[:train_num]
        val_index = data_index[train_num:]

        # Set train dataset
        train_data_x = train_x[train_index]
        train_data_y = train_y[train_index]
        train_generator = DirtyMnistGeneratorV2(train_data_x, train_data_y, batch_size, augment=True, shuffle=True)

        # Set val dataset
        val_data_x = train_x[val_index]
        val_data_y = train_y[val_index]
        val_generator = DirtyMnistGeneratorV2(val_data_x, val_data_y, batch_size, augment=False, shuffle=False)

        #######################################################
        # Training
        #######################################################
        # Set callbacks
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                      factor=0.5,
                                                      patience=10,
                                                      verbose=1,
                                                      mode='min',
                                                      min_lr=0.0001)

        save_model = keras.callbacks.ModelCheckpoint(filepath=save_model_path + '/' + model_name + '-{epoch:05d}.h5',
                                                     monitor='val_loss',
                                                     verbose=1,
                                                     save_best_only=True,
                                                     mode='min',
                                                     save_freq='epoch')

        tensorboard = keras.callbacks.TensorBoard(log_dir=log_dir, profile_batch=0)

        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                   patience=35,
                                                   verbose=1,
                                                   mode='min')

        callbacks = [reduce_lr, save_model, tensorboard, early_stop]

        try:
            # Compile Model
            with strategy.scope():
                # Get model
                model = getMultiLabelResNet50_11(input_shape, num_classes)
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
                      use_multiprocessing=False)

            # Append best model path
            best_model = glob(save_model_path + '/*')
            best_model = sorted(best_model)
            best_models.append(best_model[-1])

        except Exception as e:
            logging.error(traceback.format_exc())
            print(e)

        finally:
            # Clear Session
            keras.backend.clear_session()

            # 5개의 fold마다 가장 좋은 모델을 이용하여 예측
            # csv file path
            csv_file_path = './csv_file/' + model_name + '.csv'

            # Get result
            result = getResultOfTest(best_models, csv_file_path)

    except Exception as e:
        logging.error(traceback.format_exc())
        print(e)


if __name__ == '__main__':
    main()
