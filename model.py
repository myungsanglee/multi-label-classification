# -*- coding: utf-8 -*-

from tensorflow import keras

#######################################################
# Model function
#######################################################
def getMultiLabelEfficientNetB7(input_shape, num_labels):
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

def getMultiLabelResNet50(input_shape, num_labels):
    # Get Input tensor
    input_tensor = keras.layers.Input(shape=input_shape)

    # 1x1 convolutional layer for convert 1 channel to 3 channel
    # because input channel of EfficientNetB7 is 3
    x = keras.layers.Conv2D(filters=3, kernel_size=1, padding='same')(input_tensor)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)

    # Get backbone of ResNet50
    backbone_input_tensor = keras.layers.Input(shape=(x.shape[1], x.shape[2], x.shape[3]))
    backbone = keras.applications.ResNet50(include_top=False, weights='imagenet', input_tensor=backbone_input_tensor)
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
    model = getMultiLabelEfficientNetB7(input_shape, num_labels)
    # model = getMultiLabelResNet50(input_shape, num_labels)
    model.summary()
