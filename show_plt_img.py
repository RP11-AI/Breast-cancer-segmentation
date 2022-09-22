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
import tensorflow as tf
import keras.engine.functional
# +--------------------------------------------------------------------------------------------------------------------|


def predict(data_obj: dict[str, list], model: keras.engine.functional.Functional) -> tuple:
    img = np.array(data_obj['img'][0:10])
    mask = data_obj['mask'][0:10]
    predict_model = model.predict(img)
    return predict_model, img, mask


def plotter(img: np.array, predict_mask: np.array, ground_truth: np.array) -> None:
    plt.figure(figsize=(9, 9))
    plt.subplot(1, 3, 1), plt.imshow(img), plt.title('Ultrasound')
    plt.subplot(1, 3, 2), plt.imshow(predict_mask), plt.title('Artificial Intelligence')
    plt.subplot(1, 3, 3), plt.imshow(ground_truth), plt.title('Actual Segmentation')
    plt.show()

