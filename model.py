# -*- coding: utf-8 -*-

from tensorflow import keras
import os
import tensorflow as tf
import pandas as pd

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ''

#######################################################
# Model function
#######################################################
def getMultiLabelEfficientNetB7_01(input_shape, num_labels):
    # Get Input tensor
    input_tensor = keras.layers.Input(shape=input_shape)

    # 1x1 convolutional layer for convert 1 channel to 3 channel
    # because input channel of EfficientNetB7 is 3
    x = keras.layers.Conv2D(filters=3, kernel_size=3, padding='same')(input_tensor)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)

    # Get backbone of EfficientNetB7
    backbone_input_tensor = keras.layers.Input(shape=(x.shape[1], x.shape[2], x.shape[3]))
    backbone = keras.applications.EfficientNetB7(include_top=False, weights='imagenet', input_tensor=backbone_input_tensor)
    x = backbone(x)

    # Top
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(num_labels)(x)
    output_tensor = keras.layers.Activation('sigmoid')(x)

    return keras.Model(inputs=input_tensor, outputs=output_tensor)

def getMultiLabelEfficientNetB7_02(input_shape, num_labels):
    # Get Input tensor
    input_tensor = keras.layers.Input(shape=input_shape)

    # 1x1 convolutional layer for convert 1 channel to 3 channel
    # because input channel of EfficientNetB7 is 3
    x = keras.layers.Conv2D(filters=3, kernel_size=1, padding='same')(input_tensor)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)

    # Get backbone of EfficientNetB7
    backbone_input_tensor = keras.layers.Input(shape=(x.shape[1], x.shape[2], x.shape[3]))
    backbone = keras.applications.EfficientNetB7(include_top=False, weights='imagenet', input_tensor=backbone_input_tensor)
    x = backbone(x)

    # Top
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(num_labels)(x)
    output_tensor = keras.layers.Activation('sigmoid')(x)

    return keras.Model(inputs=input_tensor, outputs=output_tensor)

def getMultiLabelEfficientNetB7_03(input_shape, num_labels):
    # Get Input tensor
    input_tensor = keras.layers.Input(shape=input_shape)

    # 1x1 convolutional layer for convert 1 channel to 3 channel
    # because input channel of EfficientNetB7 is 3
    x = keras.layers.Conv2D(filters=3, kernel_size=3, padding='same')(input_tensor)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)

    # Get backbone of EfficientNetB7
    backbone_input_tensor = keras.layers.Input(shape=(x.shape[1], x.shape[2], x.shape[3]))
    backbone = keras.applications.EfficientNetB7(include_top=False, weights='imagenet', input_tensor=backbone_input_tensor)
    x = backbone(x)

    # Top
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(num_labels)(x)
    output_tensor = keras.layers.Activation('sigmoid')(x)

    return keras.Model(inputs=input_tensor, outputs=output_tensor)

def getMultiLabelResNet50_01(input_shape, num_labels):
    # Get Input tensor
    input_tensor = keras.layers.Input(shape=input_shape)

    # convolutional layer for convert 1 channel to 3 channel
    # because input channel of EfficientNetB7 is 3
    x = keras.layers.Conv2D(filters=3, kernel_size=3, padding='same')(input_tensor)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)

    # Get backbone of ResNet50_01
    backbone_input_tensor = keras.layers.Input(shape=(x.shape[1], x.shape[2], x.shape[3]))
    backbone = keras.applications.ResNet50(include_top=False, weights='imagenet', input_tensor=backbone_input_tensor)
    x = backbone(x)

    # Top
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(num_labels)(x)
    output_tensor = keras.layers.Activation('sigmoid')(x)

    return keras.Model(inputs=input_tensor, outputs=output_tensor)

def getMultiLabelResNet50_02(input_shape, num_labels):
    # Get Input tensor
    input_tensor = keras.layers.Input(shape=input_shape)

    # convolutional layer for convert 1 channel to 3 channel
    # because input channel of EfficientNetB7 is 3
    x = keras.layers.Conv2D(filters=3, kernel_size=3, padding='same')(input_tensor)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)

    # Get backbone of ResNet50_01
    backbone_input_tensor = keras.layers.Input(shape=(x.shape[1], x.shape[2], x.shape[3]))
    backbone = keras.applications.ResNet50(include_top=False, weights='imagenet', input_tensor=backbone_input_tensor)
    for layer in backbone.layers:
        layer.trainable = False
    x = backbone(x)

    # Top
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(num_labels)(x)
    output_tensor = keras.layers.Activation('sigmoid')(x)

    return keras.Model(inputs=input_tensor, outputs=output_tensor)

def getMultiLabelResNet50_03(input_shape, num_labels):
    # Get Input tensor
    input_tensor = keras.layers.Input(shape=input_shape)

    # convolutional layer for convert 1 channel to 3 channel
    # because input channel of EfficientNetB7 is 3
    x = keras.layers.Conv2D(filters=3, kernel_size=3, padding='same')(input_tensor)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)

    # Get backbone of ResNet50_01
    backbone_input_tensor = keras.layers.Input(shape=(x.shape[1], x.shape[2], x.shape[3]))
    backbone = keras.applications.ResNet50(include_top=False, weights='imagenet', input_tensor=backbone_input_tensor)
    backbone.trainable = True
    set_trainable = False

    for layer in backbone.layers:
        if layer.name in ['conv5_block1_1_conv']:
            set_trainable = True
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False

    x = backbone(x)

    # Top
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(num_labels)(x)
    output_tensor = keras.layers.Activation('sigmoid')(x)

    return keras.Model(inputs=input_tensor, outputs=output_tensor)

def getMultiLabelResNet50_04(input_shape, num_labels):
    # Get Input tensor
    input_tensor = keras.layers.Input(shape=input_shape)

    # convolutional layer for convert 1 channel to 3 channel
    # because input channel of EfficientNetB7 is 3
    x = keras.layers.Conv2D(filters=3, kernel_size=3, padding='same')(input_tensor)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)

    # Get backbone of ResNet50_01
    backbone_input_tensor = keras.layers.Input(shape=(x.shape[1], x.shape[2], x.shape[3]))
    backbone = keras.applications.ResNet50(include_top=False, weights=None, input_tensor=backbone_input_tensor)
    x = backbone(x)

    # Top
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(num_labels)(x)
    output_tensor = keras.layers.Activation('sigmoid')(x)

    return keras.Model(inputs=input_tensor, outputs=output_tensor)

def getMultiLabelResNet50_05(input_shape, num_labels):
    # Get Input tensor
    input_tensor = keras.layers.Input(shape=input_shape)

    # convolutional layer for convert 1 channel to 3 channel
    # because input channel of EfficientNetB7 is 3
    x = keras.layers.Conv2D(filters=3, kernel_size=3, padding='same')(input_tensor)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)

    # Get backbone of ResNet50_01
    backbone_input_tensor = keras.layers.Input(shape=(x.shape[1], x.shape[2], x.shape[3]))
    backbone = keras.applications.ResNet50(include_top=False, weights='imagenet', input_tensor=backbone_input_tensor)
    x = backbone(x)

    # Top
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(num_labels)(x)
    output_tensor = keras.layers.Activation('sigmoid')(x)

    return keras.Model(inputs=input_tensor, outputs=output_tensor)

def getMultiLabelResNet50_06(input_shape, num_labels):
    # Get Input tensor
    input_tensor = keras.layers.Input(shape=input_shape)

    # convolutional layer for convert 1 channel to 3 channel
    # because input channel of EfficientNetB7 is 3
    x = keras.layers.Conv2D(filters=3, kernel_size=3, padding='same')(input_tensor)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)

    # Get backbone of ResNet50_01
    backbone_input_tensor = keras.layers.Input(shape=(x.shape[1], x.shape[2], x.shape[3]))
    backbone = keras.applications.ResNet50(include_top=False, weights='imagenet', input_tensor=backbone_input_tensor)
    x = backbone(x)

    # Top
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(num_labels)(x)
    output_tensor = keras.layers.Activation('sigmoid')(x)

    return keras.Model(inputs=input_tensor, outputs=output_tensor)

def getMultiLabelResNet50_07(input_shape, num_labels):
    # Get Input tensor
    input_tensor = keras.layers.Input(shape=input_shape)

    # convolutional layer for convert 1 channel to 3 channel
    # because input channel of EfficientNetB7 is 3
    x = keras.layers.Conv2D(filters=3, kernel_size=3, padding='same')(input_tensor)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)

    # Get backbone of ResNet50_01
    backbone_input_tensor = keras.layers.Input(shape=(x.shape[1], x.shape[2], x.shape[3]))
    backbone = keras.applications.ResNet50(include_top=False, weights='imagenet', input_tensor=backbone_input_tensor)
    x = backbone(x)

    # Top
    x = keras.layers.GlobalAveragePooling2D()(x)

    # FC layer
    x = keras.layers.Dense(1024)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)

    x = keras.layers.Dense(num_labels)(x)
    output_tensor = keras.layers.Activation('sigmoid')(x)

    return keras.Model(inputs=input_tensor, outputs=output_tensor)

def getMultiLabelResNet50_08(input_shape, num_labels):
    # Get Input tensor
    input_tensor = keras.layers.Input(shape=input_shape)

    # convolutional layer for convert 1 channel to 3 channel
    # because input channel of EfficientNetB7 is 3
    x = keras.layers.Conv2D(filters=3, kernel_size=3, padding='same')(input_tensor)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)

    # Get backbone of ResNet50_01
    backbone_input_tensor = keras.layers.Input(shape=(x.shape[1], x.shape[2], x.shape[3]))
    backbone = keras.applications.ResNet50(include_top=False, weights='imagenet', input_tensor=backbone_input_tensor)
    x = backbone(x)

    # Top
    x = keras.layers.GlobalAveragePooling2D()(x)

    # FC layer
    x = keras.layers.Dense(1024)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)

    # FC layer
    x = keras.layers.Dense(1024)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)

    x = keras.layers.Dense(num_labels)(x)
    output_tensor = keras.layers.Activation('sigmoid')(x)

    return keras.Model(inputs=input_tensor, outputs=output_tensor)

def getMultiLabelResNet50_09(input_shape, num_labels):
    # Get Input tensor
    input_tensor = keras.layers.Input(shape=input_shape)

    # convolutional layer for convert 1 channel to 3 channel
    # because input channel of EfficientNetB7 is 3
    x = keras.layers.Conv2D(filters=3, kernel_size=1, padding='same')(input_tensor)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)

    # Get backbone of ResNet50_01
    backbone_input_tensor = keras.layers.Input(shape=(x.shape[1], x.shape[2], x.shape[3]))
    backbone = keras.applications.ResNet50(include_top=False, weights='imagenet', input_tensor=backbone_input_tensor)
    x = backbone(x)

    # Top
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(num_labels)(x)
    output_tensor = keras.layers.Activation('sigmoid')(x)

    return keras.Model(inputs=input_tensor, outputs=output_tensor)

def getMultiLabelResNet50_10(input_shape, num_labels):
    # Get Input tensor
    input_tensor = keras.layers.Input(shape=input_shape)

    # Get backbone of ResNet50_01
    backbone = keras.applications.ResNet50(include_top=False, weights='imagenet', input_tensor=input_tensor)

    # Top
    x = keras.layers.GlobalAveragePooling2D()(backbone.output)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(num_labels)(x)
    output_tensor = keras.layers.Activation('sigmoid')(x)

    return keras.Model(inputs=input_tensor, outputs=output_tensor)

def getMultiLabelResNet50V2_01(input_shape, num_labels):
    # Get Input tensor
    input_tensor = keras.layers.Input(shape=input_shape)

    # convolutional layer for convert 1 channel to 3 channel
    # because input channel of EfficientNetB7 is 3
    x = keras.layers.Conv2D(filters=3, kernel_size=3, padding='same')(input_tensor)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)

    # Get backbone of ResNet50_01
    backbone_input_tensor = keras.layers.Input(shape=(x.shape[1], x.shape[2], x.shape[3]))
    backbone = keras.applications.ResNet50V2(include_top=False, weights='imagenet', input_tensor=backbone_input_tensor)
    x = backbone(x)

    # Top
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(num_labels)(x)
    output_tensor = keras.layers.Activation('sigmoid')(x)

    return keras.Model(inputs=input_tensor, outputs=output_tensor)

def getMultiLabelEfficientNetB5_01(input_shape, num_labels):
    # Get Input tensor
    input_tensor = keras.layers.Input(shape=input_shape)

    # convolutional layer for convert 1 channel to 3 channel
    # because input channel of EfficientNetB7 is 3
    x = keras.layers.Conv2D(filters=3, kernel_size=3, padding='same')(input_tensor)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)

    # Get backbone of ResNet50_01
    backbone_input_tensor = keras.layers.Input(shape=(x.shape[1], x.shape[2], x.shape[3]))
    backbone = keras.applications.EfficientNetB5(include_top=False, weights='imagenet', input_tensor=backbone_input_tensor)
    x = backbone(x)

    # Top
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(num_labels)(x)
    output_tensor = keras.layers.Activation('sigmoid')(x)

    return keras.Model(inputs=input_tensor, outputs=output_tensor)

if __name__ == '__main__':
    input_shape = (256, 256, 1)
    num_labels = 26
    # model = getMultiLabelEfficientNetB7_01(input_shape, num_labels)
    # model = getMultiLabelResNet50(input_shape, num_labels)
    # model = testModel(input_shape, num_labels)
    # model = getMultiLabelResNet50_01(input_shape, num_labels)
    # model = getMultiLabelResNet50_02(input_shape, num_labels)
    # model = getMultiLabelResNet50_03(input_shape, num_labels)
    # model = getMultiLabelResNet50_04(input_shape, num_labels)
    # model = getMultiLabelResNet50_05(input_shape, num_labels)
    # model = getMultiLabelResNet50_06(input_shape, num_labels)
    # model = getMultiLabelResNet50_07(input_shape, num_labels)
    # model = getMultiLabelResNet50_08(input_shape, num_labels)
    # model = getMultiLabelResNet50_09(input_shape, num_labels)
    # model = getMultiLabelResNet50_10(input_shape, num_labels)
    model = getMultiLabelResNet50V2_01(input_shape, num_labels)
    # model = getMultiLabelEfficientNetB5_01(input_shape, num_labels)
    # model = keras.models.load_model('./saved_models/EfficientNetB7/test_1/EfficientNetB7-1-fold-00046.h5')
    model.summary()

    # # Checking trainable layer
    # layers = [(layer, layer.name, layer.trainable) for layer in model.layers]
    # print(layers)
    # # Find backbone layer index
    # backbone_layer_index = 0
    # for i, (_, name, _) in enumerate(layers):
    #     if name == 'efficientnetb7':
    #         backbone_layer_index = i
    #         break
    # print(backbone_layer_index)
    #
    # # Get backbone layer
    # backbone_layer = layers[backbone_layer_index][0]
    # backbone_layers = [(layer, layer.name, layer.trainable) for layer in backbone_layer.layers]
    #
    # del layers[backbone_layer_index]
    #
    # for i, layer in enumerate(backbone_layers):
    #     layers.insert(backbone_layer_index + i, layer)
    #
    # print(layers)
    #
    # meta_pd = pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])
    # meta_pd.to_csv('./layer_trainable.csv')
    # print(meta_pd)
