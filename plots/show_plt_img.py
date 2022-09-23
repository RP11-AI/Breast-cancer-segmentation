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
import keras.engine.functional
import cv2
from PIL import Image
# +--------------------------------------------------------------------------------------------------------------------|


def predict(data_obj: dict[str, list], model: keras.engine.functional.Functional) -> tuple:
    img = np.array(data_obj['img'][0:10])
    mask = np.array(data_obj['mask'][0:10])
    predict_model = model.predict(img)
    return predict_model, img, mask


def plotter(img: np.array, predict_mask: np.array, ground_truth: np.array) -> None:

    plt.figure(figsize=(10, 10))

    plt.subplot(3, 3, 1), plt.imshow(img[0]), plt.title('Ultrasound')

    plt.subplot(3, 3, 2)
    plt.imshow(img[0], interpolation='none')
    plt.imshow(predict_mask[0], interpolation='none', alpha=0.3)
    plt.title('Artificial Intelligence')

    plt.subplot(3, 3, 3)
    plt.imshow(img[0], interpolation='none')
    plt.imshow(ground_truth[0], interpolation='none', alpha=0.3)
    plt.title('Actual Segmentation')

    # New Line
    plt.subplot(3, 3, 4), plt.imshow(img[1]), plt.title('Ultrasound')

    plt.subplot(3, 3, 5)
    plt.imshow(img[1], interpolation='none')
    plt.imshow(predict_mask[1], interpolation='none', alpha=0.3)
    plt.title('Artificial Intelligence')

    plt.subplot(3, 3, 6)
    plt.imshow(img[1], interpolation='none')
    plt.imshow(ground_truth[1], interpolation='none', alpha=0.3)
    plt.title('Actual Segmentation')

    # New Line
    plt.subplot(3, 3, 7), plt.imshow(img[2]), plt.title('Ultrasound')

    plt.subplot(3, 3, 8)
    plt.imshow(img[2], interpolation='none')
    plt.imshow(predict_mask[2], interpolation='none', alpha=0.3)
    plt.title('Artificial Intelligence')

    plt.subplot(3, 3, 9)
    plt.imshow(img[2], interpolation='none')
    plt.imshow(ground_truth[2], interpolation='none', alpha=0.3)
    plt.title('Actual Segmentation')

    plt.show()


