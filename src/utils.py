import pandas as pd
import numpy as np
from skimage import color, morphology
from constant import PATH, TRAIN_CSV, VAL_CSV, TEST_CSV


def get_chestxray_from_csv():
    result = []
    for f in [PATH/TRAIN_CSV, PATH/VAL_CSV, PATH/TEST_CSV]:
        df = pd.read_csv(f, sep=' ', header=None)
        images = df.iloc[:, 0].values
        labels = df.iloc[:, 1:].values
        result.append((images, labels))
    return result

def sigmoid_np(x):
    return 1. / (1. + np.exp(-x))

def blend_segmentation(image, mask, gt_mask=None, boundary=False, alpha=1):
    h, w = image.size
    color_mask = np.zeros((h, w, 3)) # PIL Image
    if boundary: mask = morphology.dilation(mask, morphology.disk(3)) - mask
    color_mask[mask==1] = [1, 0, 0] # RGB

    if gt_mask is not None:
        gt_boundary = morphology.dilation(gt_mask, morphology.disk(3)) - gt_mask
        color_mask[gt_boundary==1] = [0, 1, 0] # RGB

    image_hsv = color.rgb2hsv(image)
    color_mask_hsv = color.rgb2hsv(color_mask)

    image_hsv[..., 0] = color_mask_hsv[..., 0]
    image_hsv[..., 1] = color_mask_hsv[..., 1] * alpha

    return color.hsv2rgb(image_hsv)
