# +--------------------------------------------------------------------------------------------------------------------+
# |                                                                                                  (module) main.py ||
# |                                                                                             Author:Pauliv, Rômulo ||
# |                                                                                      Copyright 2022, RP11.AI Ltd. ||
# |                                                                                                romulopauliv@bk.ru ||
# |                                                                                                   Encoding: UTF-8 ||
# +--------------------------------------------------------------------------------------------------------------------+

# imports -------------------------------------------------------------------------------------------------------------|
from plots.show_plt_img import predict, plotter
from data.load_data import LoadData
import tensorflow as tf
# +--------------------------------------------------------------------------------------------------------------------|

frame_obj_train = {
    'img': [],
    'mask': []
}
frame_obj_train = LoadData(
    frame_obj_train,
    img_path='data/Dataset_BUSI_with_GT/malignant',
    mask_path='data/Dataset_BUSI_with_GT/malignant'
)


model_id = tf.keras.models.load_model(filepath='AI/BreastCancerSegmentor.h5')
predict_mask, real_img, real_mask = predict(frame_obj_train, model_id)
plotter(real_img, predict_mask, real_mask)
