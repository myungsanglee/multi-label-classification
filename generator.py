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
class DirtyMnistGenerator(keras.utils.Sequence):
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
        # print(label_index)
        x, y = self.__data_gen(label_index, indexes)
        # print(x.shape, y.shape)
        # print(x.dtype, y.dtype)
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
            img = img[..., np.newaxis]
            # print(img.shape)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # img = cv2.resize(img, (self.input_shape[0], self.input_shape[1]))
            # print(img.shape)
            # img = img.astype(np.float32) / 255.
            batch_images[i] = img

            # Get label and Append data
            cls = self.meta_df.iloc[indexes[i], 1:].values.astype('float')
            # cls = keras.utils.to_categorical(cls, num_classes=self.num_classes)
            batch_class[i] = cls

        # augment images
        if self.augment:
            batch_images = self.augmenter(batch_images)

        # Convert images data type and normalization
        batch_images = batch_images.astype(np.float32) / 255.

        return batch_images, batch_class


if __name__ == '__main__':
    train_csv_path = '/home/fssv2/dirty_mnist_dataset/dirty_mnist_2nd_answer.csv'
    test_csv_path = '/home/fssv2/dirty_mnist_dataset/sample_submission.csv'
    png_path = '/home/fssv2/dirty_mnist_dataset/dirty_mnist_2nd'

    dirty_mnist_answer = pd.read_csv(train_csv_path)
    sample_submission = pd.read_csv(test_csv_path)

    a = dirty_mnist_answer.iloc[[100, 1, 2, 3, 4]]
    print(a)

    input_shape = (256, 256, 1)
    num_classes = 26
    batch_size = 1
    gen = DirtyMnistGenerator(png_path, a, input_shape, num_classes, batch_size, augment=True, shuffle=False)

    for i in range(gen.__len__()):
        batch_x, batch_y = gen.__getitem__(i)
        for i in range(batch_size):
            print(batch_y[i])
            # print(batch_x[i])

            img = batch_x[i]
            # img = cv2.cvtColor(img, cv2.C)
            cv2.imshow('test', img)
            cv2.waitKey(0)
            break

    cv2.destroyAllWindows()