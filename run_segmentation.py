# run_segmentation.py

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import json
import requests
from io import BytesIO

import time

import numpy as np
import cv2

import tensorflow as tf

from . import function
from django.conf import settings


# get best weight & class_id
def check_model_path(model_path):
    latest = tf.train.latest_checkpoint(os.path.join(model_path, 'epochs'))
    if latest == None:
        raise Exception("There isn't model check point. You should check ")
    else:
        model_info = os.path.basename(model_path).split('_')
        class_id = model_info[1]
        lr = model_info[4]
    return latest, class_id, lr

# load model
def load_model(latest, lr):
    model = function.build_model(lr)
    # -----------------------------------------
    model.load_weights(latest)
    return model


def overlay(image, mask, color=(0, 0, 200), alpha=0.5, resize=None):
    """Combines image and its segmentation mask into a single image.
    https://www.kaggle.com/code/purplejester/showing-samples-with-segmentation-mask-overlay

    Params:
        image: Training image. np.ndarray,
        mask: Segmentation mask. np.ndarray,
        color: Color for segmentation mask rendering.  tuple[int, int, int] = (255, 0, 0)
        alpha: Segmentation mask's transparency. float = 0.5,
        resize: If provided, both image and its mask are resized before blending them together.
        tuple[int, int] = (1024, 1024))

    Returns:
        image_combined: The combined image. np.ndarray

    """
    color = color[::-1] # for using openCV
    colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
    colored_mask = np.moveaxis(colored_mask, 0, -1)
    masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
    image_overlay = masked.filled()

    if resize is not None:
        image = cv2.resize(image.transpose(1, 2, 0), resize)
        image_overlay = cv2.resize(image_overlay.transpose(1, 2, 0), resize)

    image_combined = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0) # ��???���� 0??? 'Alpha Blending'??? ?????? ����??? ???�߿� ??? ????????? ??????����

    return image_combined


CLS_THRESHOLD = 0.03
CONF_THRESHOLD = 0.7
IMAGE_SIZE = (224, 224)

MODEL_NAMES = ['A1', 'A3']
model_A1_path = os.path.join(settings.MODEL_ROOT, 'U-Net_A1_200_8_0.00049974_(224, 224, 3)_(224, 224, 2)')
model_A3_path = os.path.join(settings.MODEL_ROOT, 'U-Net_A3_200_8_0.00012834_(224, 224, 3)_(224, 224, 2)')
MODEL_PATHS = [model_A1_path, model_A3_path]

def run_segm_model(img, img_vis):
    models = []
    for model_path in MODEL_PATHS:
        # Load model info
        latest, _, lr = check_model_path(model_path) # MODEL+PATHS?? ??????���??? ???????????? ??? ??? ???????? ?????????
        # Build mode & load trained weight
        model = load_model(latest, lr)
        models.append(model)

    pred_clses = []
    pred_masks = []
    for model in models:
        t0 = time.time()  # temp to measure time
        pred = model.predict(img[tf.newaxis, ...])[0]  # tf.newaxis: make batch dim, [0]: remove batch dim
        pred_mask = pred[:, :, 1]
        print(time.time() - t0)  # temp to measure time

        pred_valid = pred_mask > CONF_THRESHOLD
        pred_mask[~pred_valid] = 0.  # ignore invalid predictions

        # Do classification
        if pred_valid.sum() / (224 * 244) > CLS_THRESHOLD - 0.025:
            pred_clses.append(True)
        else:
            pred_clses.append(False)
        pred_masks.append(pred_mask)

    img_overlays = [] # List to store generated img_overlay images
    for model_name, pred_mask in zip(MODEL_NAMES, pred_masks):
        img_overlay = overlay(img_vis, pred_mask, color=(200, 0, 0), alpha=0.3)
        img_overlays.append(img_overlay)
    
    return img_overlays, pred_clses
