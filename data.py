import pandas as pd
import numpy as np
import os
import sys
import cv2
from tqdm import tqdm
import tensorflow as tf
from sklearn.model_selection import KFold

#######################################################
# Set GPU
#######################################################
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
# GPU를 아예 못 보게 하려면: os.environ["CUDA_VISIBLE_DEVICES"]=''
# GPU 0만 보게 하려면: os.environ["CUDA_VISIBLE_DEVICES"]='0'
# GPU 1만 보게 하려면: os.environ["CUDA_VISIBLE_DEVICES"]='0'
# GPU 0과 1을 보게 하려면: os.environ["CUDA_VISIBLE_DEVICES"]='0,1'

def getDatasetArray(png_path, meta_df, input_shape, num_classes):
    # return parameters
    dst_x = np.zeros(shape=(len(meta_df), input_shape[0], input_shape[1], input_shape[2]), dtype=np.uint8)
    dst_y = np.zeros(shape=(len(meta_df), num_classes), dtype=np.float32)

    indexes = np.arange(len(meta_df))
    label_indexes = meta_df.iloc[indexes, 0].values
    for i, label_index in tqdm(enumerate(label_indexes)):
        img_path = os.path.join(png_path, str(label_index).zfill(5) + '.png')
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = np.expand_dims(img, axis=-1)
        dst_x[i] = img

        label = meta_df.iloc[indexes[i], 1:].values.astype(np.float32)
        dst_y[i] = label

    return dst_x, dst_y

def getDatasetArrayOfColor(png_path, meta_df, input_shape, num_classes):
    # return parameters
    dst_x = np.zeros(shape=(len(meta_df), input_shape[0], input_shape[1], input_shape[2]), dtype=np.uint8)
    dst_y = np.zeros(shape=(len(meta_df), num_classes), dtype=np.float32)

    indexes = np.arange(len(meta_df))
    label_indexes = meta_df.iloc[indexes, 0].values
    for i, label_index in tqdm(enumerate(label_indexes)):
        img_path = os.path.join(png_path, str(label_index).zfill(5) + '.png')
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        dst_x[i] = img

        label = meta_df.iloc[indexes[i], 1:].values.astype(np.float32)
        dst_y[i] = label

    return dst_x, dst_y

if __name__ == '__main__':

    train_csv_path = '/home/fssv2/dirty_mnist_dataset/dirty_mnist_2nd_answer.csv'
    test_csv_path = '/home/fssv2/dirty_mnist_dataset/sample_submission.csv'
    train_png_path = '/home/fssv2/dirty_mnist_dataset/dirty_mnist_2nd'
    test_png_path = '/home/fssv2/dirty_mnist_dataset/test_dirty_mnist_2nd'

    dirty_mnist_answer = pd.read_csv(train_csv_path)
    sample_submission = pd.read_csv(test_csv_path)

    ###############################
    # Save RGB Data
    ###############################
    # input_shape = (256, 256, 3)
    # num_classes = 26
    #
    # train_x, train_y = getDatasetArrayOfColor(train_png_path, dirty_mnist_answer, input_shape, num_classes)
    # np.save('./data_array_color/train_x.npy', train_x)
    # np.save('./data_array_color/train_y.npy', train_y)
    # print(sys.getsizeof(train_x) * 1e-9)
    # print(sys.getsizeof(train_y) * 1e-9)
    #
    # test_x, test_y = getDatasetArrayOfColor(test_png_path, sample_submission, input_shape, num_classes)
    # np.save('./data_array_color/test_x.npy', test_x)
    # np.save('./data_array_color/test_y.npy', test_y)
    # print(sys.getsizeof(test_x) * 1e-9)
    # print(sys.getsizeof(test_y) * 1e-9)

    ###############################
    # Save Gray Data
    ###############################
    # input_shape = (256, 256, 1)
    # num_classes = 26
    #
    # train_x, train_y = getDatasetArray(train_png_path, dirty_mnist_answer, input_shape, num_classes)
    # np.save('./data_array_gray/train_x.npy', train_x)
    # np.save('./data_array_gray/train_y.npy', train_y)
    # print(sys.getsizeof(train_x) * 1e-9)
    # print(sys.getsizeof(train_y) * 1e-9)
    #
    # test_x, test_y = getDatasetArray(test_png_path, sample_submission, input_shape, num_classes)
    # np.save('./data_array_gray/test_x.npy', test_x)
    # np.save('./data_array_graytest_y.npy', test_y)
    # print(sys.getsizeof(test_x) * 1e-9)
    # print(sys.getsizeof(test_y) * 1e-9)
