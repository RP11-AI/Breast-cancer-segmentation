# +--------------------------------------------------------------------------------------------------------------------+
# |                                                                                                      load_data.py ||
# |                                                                                             Author:Pauliv, Rômulo ||
# |                                                                                      Copyright 2022, RP11.AI Ltd. ||
# |                                                                                                romulopauliv@bk.ru ||
# |                                                                                                   Encoding: UTF-8 ||
# +--------------------------------------------------------------------------------------------------------------------+

# imports -------------------------------------------------------------------------------------------------------------|
import cv2
import os
from typing import Union
import matplotlib.pyplot as plt
import logs.log as log
# +--------------------------------------------------------------------------------------------------------------------|


def LoadData(
        frame_obj: Union[dict[str, list], None] = None,
        img_path: Union[str, None] = None,
        mask_path: Union[str, None] = None,
        shape: int = 256) -> dict[str, list]:
    """
    Load and organize images in memory.
    param frame_obj: Dictionary configured to append arrays to img and mask keys.
    param img_path: Ultrasound images directory.
    param mask_path: Ultrasound mask directory.
    param shape: Square resize size in pixels.
    """
    img_names = os.listdir(img_path)
    names: list = []
    mask_names: list = []
    u_names: list = []

    for i in img_names:
        u_names.append(i.split(')')[0])

    u_names = list(set(u_names))

    for i in u_names:
        names.append(i + ').png')
        mask_names.append(i + ')_mask.png')

    img_add_r = img_path + '/'
    mask_add_r = mask_path + '/'

    for i in range(len(names)):
        log.image_reading(names[i])
        img = plt.imread(img_add_r + names[i])
        log.ultra_sound_confirm()
        mask = plt.imread(mask_add_r + mask_names[i])
        log.mask_confirm()

        img = cv2.resize(img, (shape, shape))
        mask = cv2.resize(mask, (shape, shape))

        frame_obj['img'].append(img)
        frame_obj['mask'].append(mask)

    return frame_obj
