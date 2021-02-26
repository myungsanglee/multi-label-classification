#-*- coding: utf-8 -*-

import cv2
from tensorflow import keras
from imgaug import augmenters as iaa
import numpy as np
import pandas as pd
import os

#######################################################
# Generator class
#######################################################
class DirtyMnistGeneratorV1(keras.utils.Sequence):
    def __init__(self, dir_path, meta_df, input_shape, num_classes, batch_size, augment=False, shuffle=False):

        self.dir_path = dir_path
        self.meta_df = meta_df
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.augment = augment
        self.indexes = None
        self.on_epoch_end()

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.meta_df))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.floor(len(self.meta_df) / self.batch_size)) #데이터를 Batch_size만큼 나누기

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size] #범위 설정
        label_index = self.meta_df.iloc[indexes, 0].values # Get index of label (for get image)
        x, y = self.__data_gen(label_index, indexes)
        return x, y

    def augmenter(self, images):
        seq = iaa.Sequential(
            [
                iaa.SomeOf((0, 7),
                           [
                               iaa.Identity(),
                               iaa.Rotate(),
                               iaa.Posterize(),
                               iaa.Sharpen(),
                               iaa.TranslateX(),
                               iaa.GammaContrast(),
                               iaa.Solarize(),
                               iaa.ShearX(),
                               iaa.TranslateY(),
                               iaa.HistogramEqualization(),
                               # iaa.MultiplyHueAndSaturation(),
                               # iaa.MultiplyAndAddToBrightness(),
                               iaa.ShearY(),
                               iaa.ScaleX(),
                               iaa.ScaleY(),
                               iaa.Rot90(k=(1, 3))
                           ]
                           )
            ]
        )
        return seq.augment_images(images)

    def __data_gen(self, label_indexes, indexes):
        cv2.setNumThreads(0)
        batch_images = np.zeros(shape=(self.batch_size, self.input_shape[0], self.input_shape[1], self.input_shape[2]),
                                dtype=np.uint8)

        batch_class = np.zeros(shape=(self.batch_size, self.num_classes), dtype=np.float32)

        for i, label_index in enumerate(label_indexes):
            # Get img and Append data
            img_path = os.path.join(self.dir_path, str(label_index).zfill(5) + '.png')
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = np.expand_dims(img, axis=-1)
            batch_images[i] = img

            # Get label and Append data
            cls = self.meta_df.iloc[indexes[i], 1:].values.astype('float')
            batch_class[i] = cls

        # augment images
        if self.augment:
            batch_images = self.augmenter(batch_images)

        # Convert images data type and normalization
        batch_images = batch_images.astype(np.float32) / 255.

        return batch_images, batch_class

class DirtyMnistGeneratorV2(keras.utils.Sequence):
    def __init__(self, data_x, data_y, batch_size, augment=False, shuffle=False):

        self.data = data_x
        self.label = data_y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.indexes = None
        self.on_epoch_end()

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.data))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.floor(len(self.data) / self.batch_size)) #데이터를 Batch_size만큼 나누기

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size] #범위 설정
        data = self.data[indexes]
        label = self.label[indexes]

        if self.augment:
            data = self.augmenter(data)
        data = data.astype(np.float32) / 255.

        return data, label

    def augmenter(self, images):
        seq = iaa.Sequential(
            [
                iaa.SomeOf((0, 2),
                           [
                               iaa.Identity(),
                               # iaa.Rotate(),
                               iaa.Solarize(threshold=(0, 5)),
                               iaa.Sharpen(alpha=(0.9, 1.), lightness=1),
                               iaa.HistogramEqualization(),
                               # iaa.TranslateX(),
                               # iaa.Posterize(),
                               # iaa.GammaContrast(),
                               # iaa.ShearX(),
                               # iaa.TranslateY(),
                               # iaa.MultiplyHueAndSaturation(),
                               # iaa.MultiplyAndAddToBrightness(),
                               # iaa.ShearY(),
                               # iaa.ScaleX()
                               # iaa.ScaleY()
                               iaa.Rot90(k=(1, 3))
                           ]
                           )
            ]
        )
        return seq.augment_images(images)

if __name__ == '__main__':
    ############################
    # V1
    ############################
    # train_csv_path = '/home/fssv2/dirty_mnist_dataset/dirty_mnist_2nd_answer.csv'
    # test_csv_path = '/home/fssv2/dirty_mnist_dataset/sample_submission.csv'
    # train_png_path = '/home/fssv2/dirty_mnist_dataset/dirty_mnist_2nd'
    # test_png_path = '/home/fssv2/dirty_mnist_dataset/test_dirty_mnist_2nd'
    #
    # dirty_mnist_answer = pd.read_csv(train_csv_path)
    # sample_submission = pd.read_csv(test_csv_path)
    #
    # input_shape = (256, 256, 1)
    # num_classes = 26
    # batch_size = 1
    # gen = DirtyMnistGeneratorV1(test_png_path, sample_submission, input_shape, num_classes, batch_size, augment=False, shuffle=False)
    #
    # for i in range(gen.__len__()):
    #     batch_x, batch_y = gen.__getitem__(i)
    #     for j in range(batch_size):
    #         print(batch_y[j])
    #         # print(batch_x[i])
    #
    #         img = batch_x[j]
    #         # img = cv2.cvtColor(img, cv2.C)
    #         cv2.imshow('test', img)
    #         cv2.waitKey(0)
    #         break
    #
    # cv2.destroyAllWindows()

    ############################
    # V2
    ############################
    # Gray data
    train_x = np.load('./data_array_gray/train_x.npy')
    train_y = np.load('./data_array_gray/train_y.npy')

    # RBG data
    # train_x = np.load('./data_array_color/train_x.npy')
    # train_y = np.load('./data_array_color/train_y.npy')

    batch_size = 32
    train_generator = DirtyMnistGeneratorV2(train_x, train_y, batch_size, augment=True, shuffle=False)

    for i in range(train_generator.__len__()):
        x, y = train_generator.__getitem__(i)
        print(x.shape, x.dtype)
        print(y.shape, y.dtype)
        flag = False
        for j in range(batch_size):
            img = x[j]
            label = y[j]
            print('label: {}'.format(label))
            cv2.imshow('Test', img)
            key = cv2.waitKey(0)
            if key == 27:
                flag = True
                break

        if flag:
            break