# +--------------------------------------------------------------------------------------------------------------------+
# |                                                                                               model_definition.py ||
# |                                                                                             Author:Pauliv, Rômulo ||
# |                                                                                      Copyright 2022, RP11.AI Ltd. ||
# |                                                                                                romulopauliv@bk.ru ||
# |                                                                                                   Encoding: UTF-8 ||
# +--------------------------------------------------------------------------------------------------------------------+

# imports -------------------------------------------------------------------------------------------------------------|
import tensorflow as tf
# +--------------------------------------------------------------------------------------------------------------------|


def CONV2dBLOCK(
        inputTensor,
        numFilters: int,
        kernelSize: int = 3,
        doBatchNorm: bool = True
):
    # First Conv
    x = tf.keras.layers.Conv2D(
        filters=numFilters,
        kernel_size=(kernelSize, kernelSize),
        kernel_initializer='he_normal',
        padding='same')(inputTensor)

    if doBatchNorm:
        x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Activation('relu')(x)

    # Second Conv
    x = tf.keras.layers.Conv2D(
        filters=numFilters,
        kernel_size=(kernelSize, kernelSize),
        kernel_initializer='he_normal',
        padding='same')(x)

    if doBatchNorm:
        x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Activation('relu')(x)

    return x


# Now Defining Unet
def GiveMeUnet(
        inputImage,
        numFilters: int = 16,
        dropOuts: float = 0.1,
        doBatchNorm: bool = True
):
    c1 = CONV2dBLOCK(inputImage, numFilters * 1, kernelSize=3, doBatchNorm=doBatchNorm)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)
    p1 = tf.keras.layers.Dropout(dropOuts)(p1)

    c2 = CONV2dBLOCK(p1, numFilters * 2, kernelSize=3, doBatchNorm=doBatchNorm)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
    p2 = tf.keras.layers.Dropout(dropOuts)(p2)

    c3 = CONV2dBLOCK(p2, numFilters * 4, kernelSize=3, doBatchNorm=doBatchNorm)
    p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)
    p3 = tf.keras.layers.Dropout(dropOuts)(p3)

    c4 = CONV2dBLOCK(p3, numFilters * 8, kernelSize=3, doBatchNorm=doBatchNorm)
    p4 = tf.keras.layers.MaxPooling2D((2, 2))(c4)
    p4 = tf.keras.layers.Dropout(dropOuts)(p4)

    c5 = CONV2dBLOCK(p4, numFilters * 16, kernelSize=3, doBatchNorm=doBatchNorm)

    # defining decoder path
    u6 = tf.keras.layers.Conv2DTranspose(numFilters * 8, (3, 3), strides=(2, 2), padding='same')(c5)
    u6 = tf.keras.layers.concatenate([u6, c4])
    u6 = tf.keras.layers.Dropout(dropOuts)(u6)
    c6 = CONV2dBLOCK(u6, numFilters * 8, kernelSize=3, doBatchNorm=doBatchNorm)

    u7 = tf.keras.layers.Conv2DTranspose(numFilters * 4, (3, 3), strides=(2, 2), padding='same')(c6)
    u7 = tf.keras.layers.concatenate([u7, c3])
    u7 = tf.keras.layers.Dropout(dropOuts)(u7)
    c7 = CONV2dBLOCK(u7, numFilters * 4, kernelSize=3, doBatchNorm=doBatchNorm)

    u8 = tf.keras.layers.Conv2DTranspose(numFilters * 2, (3, 3), strides=(2, 2), padding='same')(c7)
    u8 = tf.keras.layers.concatenate([u8, c2])
    u8 = tf.keras.layers.Dropout(dropOuts)(u8)
    c8 = CONV2dBLOCK(u8, numFilters * 2, kernelSize=3, doBatchNorm=doBatchNorm)

    u9 = tf.keras.layers.Conv2DTranspose(numFilters * 1, (3, 3), strides=(2, 2), padding='same')(c8)
    u9 = tf.keras.layers.concatenate([u9, c1])
    u9 = tf.keras.layers.Dropout(dropOuts)(u9)
    c9 = CONV2dBLOCK(u9, numFilters * 1, kernelSize=3, doBatchNorm=doBatchNorm)

    output = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = tf.keras.Model(inputs=[inputImage], outputs=[output])
    return model


