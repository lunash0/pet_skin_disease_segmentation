import os
import requests
from io import BytesIO

import time

import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf

import function


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
    color = color[::-1]
    colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
    colored_mask = np.moveaxis(colored_mask, 0, -1)
    masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
    image_overlay = masked.filled()

    if resize is not None:
        image = cv2.resize(image.transpose(1, 2, 0), resize)
        image_overlay = cv2.resize(image_overlay.transpose(1, 2, 0), resize)

    image_combined = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)

    return image_combined


CLS_THRESHOLD = 0.03
CONF_THRESHOLD = 0.7
IMAGE_SIZE = (224, 224)
# MODEL_PATH = "./workspace/data/dog/result_segm/U-Net_A1_200_8_0.00049974_(224, 224, 3)_(224, 224, 2)"  # A1 model path

MODEL_NAMES = ['A1', 'A3']
MODEL_PATHS = [
    # "./workspace/data/dog/result_segm/U-Net_A1_200_8_0.00049974_(224, 224, 3)_(224, 224, 2)",  # A1 model path
    "./workspace/data/dog/result_segm/U-Net_A3_200_8_0.00012834_(224, 224, 3)_(224, 224, 2)"  # A3 model path
]


if __name__ == "__main__":

    models = []
    for model_path in MODEL_PATHS:
        # Load model info
        latest, class_id, lr = check_model_path(model_path)

        # Build mode & load trained weight
        model = load_model(latest, lr)
        models.append(model)

    # while True:
    # TODO: AWS 서버에 이미지 업로드 되었는지 체크하는 코드 작성

    # TODO: if new_data_arrived:
    #           run code
    #       else:
    #           wait

    # Load image form local (Temporary)
    img = Image.open('./test/aihub_tmp1.jpg')

    # Get image from server
    # url = 'https://mblogthumb-phinf.pstatic.net/MjAxODEwMDNfMjE2/MDAxNTM4NTUwNzUwOTEx.mMEklChU0H7zC-A41Ti478q53ZT4Qnr0ZZp8hUdhaiog.TuL41isBADprznlsD58MP2oL3ZG9GmHVa8w8rpfSEp4g.PNG.hansolpet/%EA%B0%95%EC%95%84%EC%A7%80_%ED%94%BC%EB%B6%80%EC%97%BC%EC%A6%9D.PNG?type=w800'
    # res = requests.get(url)
    # img = Image.open(BytesIO(res.content))

    # Preprocess (resize, normalization)

    #TEST
    # canvas = np.ones((224, 224, 3), dtype=np.uint8)*210
    # img = img.resize((224, 224))
    # img = np.array(img)[:180, :180]
    # canvas[:180, :180, :] = np.array(img)
    # img = canvas

    img = img.resize(IMAGE_SIZE)
    img_vis = np.array(img)  # copy image for visualization
    img = np.array(img) / 255.

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
        if pred_valid.sum() / (224 * 244) > CLS_THRESHOLD:
            pred_clses.append(True)
        else:
            pred_clses.append(False)
        pred_masks.append(pred_mask)

    for model_name, pred_mask in zip(MODEL_NAMES, pred_masks):
        img_overlay = overlay(img_vis, pred_mask, color=(200, 0, 0), alpha=0.3)

        plt.imshow(img_overlay)
        # plt.title(model_name)
        plt.show()

    # TODO: send predicted result to server

    # break

    # t=1

