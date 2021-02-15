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

if __name__ == '__main__':

    train_csv_path = '/home/fssv2/dirty_mnist_dataset/dirty_mnist_2nd_answer.csv'
    test_csv_path = '/home/fssv2/dirty_mnist_dataset/sample_submission.csv'
    train_png_path = '/home/fssv2/dirty_mnist_dataset/dirty_mnist_2nd'
    test_png_path = '/home/fssv2/dirty_mnist_dataset/test_dirty_mnist_2nd'

    dirty_mnist_answer = pd.read_csv(train_csv_path)
    sample_submission = pd.read_csv(test_csv_path)

    input_shape = (256, 256, 1)
    num_classes = 26
    batch_size = 1

    # train_x, train_y = getDatasetArray(train_png_path, dirty_mnist_answer, input_shape, num_classes)
    # np.save('./train_x.npy', train_x)
    # np.save('./train_y.npy', train_y)
    # print(sys.getsizeof(train_x) * 1e-9)
    # print(sys.getsizeof(train_y) * 1e-9)

    # test_x, test_y = getDatasetArray(test_png_path, sample_submission, input_shape, num_classes)
    # np.save('./test_x.npy', test_x)
    # np.save('./test_y.npy', test_y)
    # print(sys.getsizeof(test_x) * 1e-9)
    # print(sys.getsizeof(test_y) * 1e-9)
    # print(test_x.shape, test_x.dtype)
    # print(test_y.shape, test_y.dtype)

    # train_x = np.load('./train_x.npy')
    # train_x = train_x.astype(np.float32) / 255.
    # train_y = np.load('./train_y.npy')
    # print(sys.getsizeof(train_x) * 1e-9)
    # print(sys.getsizeof(train_y) * 1e-9)
    #
    # kfold = KFold(n_splits=10, shuffle=True, random_state=0)
    #
    # for train_index, val_index in kfold.split(train_x, train_y):
    #     print(train_index, len(train_index))
    #     print(val_index, len(val_index))
    #     train_x_dataset = train_x[train_index]
    #     train_y_dataset = train_y[train_index]
    #     print(train_x_dataset.shape, len(train_x_dataset))
    #     print(train_y_dataset.shape, len(train_y_dataset))
    #     print('')

    # train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    #
    # for x, y in train_dataset.as_numpy_iterator():
    #     print(x)
    #     print(y)
    #     break
    #
    # while True:
    #     pass