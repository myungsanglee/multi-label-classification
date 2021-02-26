#######################################################
# Import library
#######################################################
import os
from tensorflow import keras
import tensorflow as tf
from generator import DirtyMnistGeneratorV2
import pandas as pd
from tqdm import tqdm
import numpy as np
from glob import glob
import logging
import traceback
from send_mail import sendMail

logging.basicConfig(level=logging.ERROR)

#######################################################
# Set GPU
#######################################################
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

#######################################################
# Set Memory
#######################################################
# gpus = tf.config.experimental.list_physical_devices('GPU')
# print(gpus)
# if gpus:
#   try:
#     tf.config.experimental.set_virtual_device_configuration(
#         gpus[0],
#         [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
#   except RuntimeError as e:
#     print(e)

#######################################################
# Functions
#######################################################
def getResultOfTest(model_list, save_path):
    try:
        # Get Test data
        test_x = np.load('./data_array_gray/test_x.npy')
        test_y = np.load('./data_array_gray/test_y.npy')
        test_batch_size = 20
        test_generator = DirtyMnistGeneratorV2(test_x, test_y, test_batch_size, augment=False, shuffle=False)

        predictions_list = []
        for model_path in model_list:
            prediction_array = np.zeros(shape=(test_y.shape[0], test_y.shape[1]))
            model = keras.models.load_model(model_path)

            for i in tqdm(range(test_generator.__len__())):
                batch_x, batch_y = test_generator.__getitem__(i)
                inference = model(batch_x, training=False)
                inference = inference.numpy()
                inference = (inference > 0.5)
                inference = inference.astype(int)

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
        sample_submission.to_csv(save_path, index=False)

        return 1

    except Exception as e:
        sendMail(subject='getResultOfTest 함수 에러 발생',
                 contents=traceback.format_exc() + '\n' + str(e)
                 )

        logging.error(traceback.format_exc())
        print(e)
        return None

    except:
        sendMail(subject='getResultOfTest 함수 에러 발생',
                 contents=traceback.format_exc()
                 )

        logging.error(traceback.format_exc())
        return None

#######################################################
# Test generator
#######################################################
# test_x = np.load('./test_x.npy')
# test_y = np.load('./test_y.npy')
# batch_size = 20
# test_generator = DirtyMnistGeneratorV2(test_x, test_y, batch_size, augment=False, shuffle=False)

#######################################################
# Evaluate model
#######################################################
# sample_submission = pd.read_csv('/home/fssv2/dirty_mnist_dataset/sample_submission.csv')
#
# for i in tqdm(range(test_generator.__len__())):
#     batch_x, batch_y = test_generator.__getitem__(i)
#     inference = model(batch_x, training=False)
#     inference = inference.numpy()
#     inference = (inference > 0.5)
#     inference = inference.astype(int)
#
#     batch_index = batch_size * i
#     sample_submission.iloc[batch_index:batch_index + batch_size, 1:] = inference
#
# sample_submission.to_csv('./test_model.csv', index=False)


#######################################################
# Main
#######################################################
if __name__=='__main__':
    # Get model list
    model_list = []
    # folder_list = glob('./saved_models/ResNet50_05/*')
    # folder_list = sorted(folder_list)
    # print(folder_list)
    # for folder in folder_list[:]:
    #     models = glob(folder + '/*')
    #     models = sorted(models)
    #     if len(models) != 0:
    #         model_list.append(models[-1])
    # print(model_list)

    model_path = glob('./saved_models/ResNet50_01/9/*')
    model_path = sorted(model_path)
    model_path = model_path[-1]
    print(model_path)
    model_list.append(model_path)
    # Set save path
    save_path = 'csv_file/ResNet50_01_9.csv'


    # Get result
    result = getResultOfTest(model_list, save_path)
    print(result)

