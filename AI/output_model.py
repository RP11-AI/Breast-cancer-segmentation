# +--------------------------------------------------------------------------------------------------------------------+
# |                                                                                                   output_model.py ||
# |                                                                                             Author:Pauliv, Rômulo ||
# |                                                                                      Copyright 2022, RP11.AI Ltd. ||
# |                                                                                                romulopauliv@bk.ru ||
# |                                                                                                   Encoding: UTF-8 ||
# +--------------------------------------------------------------------------------------------------------------------+

# imports -------------------------------------------------------------------------------------------------------------|
import numpy as np
from data.load_data import LoadData
import tensorflow as tf
from model_definition import GiveMeUnet
import matplotlib.pyplot as plt
# +--------------------------------------------------------------------------------------------------------------------|

frame_obj_train = {
    'img': [],
    'mask': []
}

frame_obj_train = LoadData(
    frame_obj=frame_obj_train,
    img_path='Dataset_BUSI_with_GT/benign',
    mask_path='Dataset_BUSI_with_GT/benign',
)

frame_obj_train = LoadData(
    frame_obj=frame_obj_train,
    img_path='Dataset_BUSI_with_GT/malignant',
    mask_path='Dataset_BUSI_with_GT/malignant'
)


inputs = tf.keras.layers.Input((256, 256, 3))
my_transformer = GiveMeUnet(inputs, dropOuts=0.07)
my_transformer.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

retVal = my_transformer.fit(np.array(frame_obj_train['img']), np.array(frame_obj_train['mask']),
                            epochs=50, verbose=0)

plt.plot(retVal.history['loss'], label='training_loss')
plt.plot(retVal.history['accuracy'], label='training_accuracy')
plt.legend()
plt.grid(True)

my_transformer.save('BreastCancerSegmentor.h5')
