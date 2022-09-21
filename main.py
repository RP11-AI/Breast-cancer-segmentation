# +--------------------------------------------------------------------------------------------------------------------+
# |                                                                                                  (module) main.py ||
# |                                                                                             Author:Pauliv, Rômulo ||
# |                                                                                      Copyright 2022, RP11.AI Ltd. ||
# |                                                                                                romulopauliv@bk.ru ||
# |                                                                                                   Encoding: UTF-8 ||
# +--------------------------------------------------------------------------------------------------------------------+

# imports -------------------------------------------------------------------------------------------------------------|
import numpy as np
import matplotlib.pyplot as plt
from load_data import LoadData
import tensorflow as tf
# +--------------------------------------------------------------------------------------------------------------------|


def predict16(valMap, model, shape=256):
    img = valMap['img'][0:16]
    mask = valMap['mask'][0:16]
    imgProc = img[0:16]
    imgProc = np.array(img)
    predictions = model.predict(imgProc)
    return predictions, imgProc, mask


def Plotter(img, predMask, groundTruth):
    plt.figure(figsize=(9, 9))

    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title(' image')

    plt.subplot(1, 3, 2)
    plt.imshow(predMask)
    plt.title('Predicted mask')

    plt.subplot(1, 3, 3)
    plt.imshow(groundTruth)
    plt.title('Actual mask')


frame_obj_train = {
    'img': [],
    'mask': []
}


frame_obj_train = LoadData(
    frame_obj_train,
    img_path='Dataset_BUSI_with_GT/normal',
    mask_path='Dataset_BUSI_with_GT/normal'
)


model_id = tf.keras.models.load_model(filepath='BreastCancerSegmentor.h5')


six_teen_prediction, actual, masks = predict16(frame_obj_train, model=model_id)
Plotter(actual[1], six_teen_prediction[1][:, :, 0], masks[1])
Plotter(actual[2], six_teen_prediction[2][:, :, 0], masks[2])
Plotter(actual[3], six_teen_prediction[3][:, :, 0], masks[3])
Plotter(actual[4], six_teen_prediction[4][:, :, 0], masks[4])
Plotter(actual[5], six_teen_prediction[5][:, :, 0], masks[5])
Plotter(actual[6], six_teen_prediction[6][:, :, 0], masks[6])
Plotter(actual[7], six_teen_prediction[7][:, :, 0], masks[7])
plt.show()
