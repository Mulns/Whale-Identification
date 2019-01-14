# Read the dataset description
import gzip
# Read or generate p2h, a dictionary of image name to image id (picture to hash)
import pickle
import platform
import random
# Suppress annoying stderr output when importing keras.
import sys
from lap import lapjv
from math import sqrt
# Determine the size of each image
from os.path import isfile
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import read_csv
from PIL import Image as pil_image
from imagehash import phash
from scipy.ndimage import affine_transform
from tqdm import tqdm
import time
from tensorflow import keras

TRAIN_DF = '../Dataset/train.csv'
SUB_Df = '../Dataset/sample_submission.csv'
TRAIN = '../Dataset/train/'
TEST = '../Dataset/test/'
P2H = '../Dataset/metadata/p2h.pickle'
P2SIZE = '../Dataset/metadata/p2size.pickle'
H2P = "../Dataset/metadata/h2p.pickle"
W2HS = "../Dataset/metadata/w2hs.pickle"
H2WS = "../Dataset/metadata/w2hs.pickle"
W2TS = "../Dataset/metadata/w2ts.pickle"
T2I = "../Dataset/metadata/t2i.pickle"
TRAIN_ID = "../Dataset/metadata/train_id.pickle"
BB_DF = "../Dataset/metadata/bounding_boxes.csv"

META_LIST = [P2H, P2SIZE, H2P, W2HS, W2TS, H2WS, T2I, TRAIN_ID]

# tagged = dict([(p, w) for _, p, w in read_csv(TRAIN_DF).to_records()])
# submit = [p for _, p, _ in read_csv(SUB_Df).to_records()]
# join = list(tagged.keys()) + submit

# img_shape = (384, 384, 1)  # The image shape used by the model
# anisotropy = 2.15  # The horizontal compression ratio
# crop_margin = 0.05  # The margin added around the bounding box to compensate for bounding box inaccuracy


def load_meta():
    meta_dic = {}
    keys = ['p2h', 'p2size', 'h2p', 'w2hs', 'w2ts', 'h2ws', 't2i', 'train']
    for i, meta_path in enumerate(META_LIST):
        if not isfile(meta_path):
            raise ValueError(
                "Please run save_meta.py for metadata generating.")
        with open(meta_path, 'rb') as f:
            meta_dic[keys[i]] = pickle.load(f)
    p2bb = pd.read_csv(BB_DF).set_index("Image")
    meta_dic['p2bb'] = p2bb
    meta_dic['anisotropy'] = 2.15
    meta_dic['crop_margin'] = 0.05
    return meta_dic


# old_stderr = sys.stderr
# sys.stderr = open('/dev/null' if platform.system() != 'Windows' else 'nul',
#                     'w')
# sys.stderr = old_stderr


def expand_path(p):
    if isfile(TRAIN + p):
        return TRAIN + p
    if isfile(TEST + p):
        return TEST + p
    return p


def show_whale(imgs, per_row=2):
    n = len(imgs)
    rows = (n + per_row - 1) // per_row
    cols = min(per_row, n)
    _, axes = plt.subplots(
        rows, cols, figsize=(24 // per_row * cols, 24 // per_row * rows))
    for ax in axes.flatten():
        ax.axis('off')
    for _, (img, ax) in enumerate(zip(imgs, axes.flatten())):
        ax.imshow(img.convert('RGB'))


def read_raw_image(p):
    img = pil_image.open(expand_path(p))
    return img


def build_transform(rotation, shear, height_zoom, width_zoom, height_shift,
                    width_shift):
    """
    Build a transformation matrix with the specified characteristics.
    """
    rotation = np.deg2rad(rotation)
    shear = np.deg2rad(shear)
    rotation_matrix = np.array([[np.cos(rotation),
                                 np.sin(rotation), 0],
                                [-np.sin(rotation),
                                 np.cos(rotation), 0], [0, 0, 1]])
    shift_matrix = np.array([[1, 0, height_shift], [0, 1, width_shift],
                             [0, 0, 1]])
    shear_matrix = np.array([[1, np.sin(shear), 0], [0, np.cos(shear), 0],
                             [0, 0, 1]])
    zoom_matrix = np.array([[1.0 / height_zoom, 0, 0],
                            [0, 1.0 / width_zoom, 0], [0, 0, 1]])
    shift_matrix = np.array([[1, 0, -height_shift], [0, 1, -width_shift],
                             [0, 0, 1]])
    return np.dot(
        np.dot(rotation_matrix, shear_matrix), np.dot(zoom_matrix,
                                                      shift_matrix))


def read_cropped_image(p, img_shape, augment, **kwargs):
    """
    @param p : the name of the picture to read
    @param augment: True/False if data augmentation should be performed
    @return a numpy array with the transformed image
    """
    # If an image id was given, convert to filename
    if p in kwargs['h2p']:
        p = kwargs['h2p'][p]
    size_x, size_y = kwargs['p2size'][p]

    # Determine the region of the original image we want to capture based on the bounding box.
    row = kwargs['p2bb'].loc[p]
    x0, y0, x1, y1 = row['x0'], row['y0'], row['x1'], row['y1']
    dx = x1 - x0
    dy = y1 - y0
    x0 -= dx * kwargs['crop_margin']
    x1 += dx * kwargs['crop_margin'] + 1
    y0 -= dy * kwargs['crop_margin']
    y1 += dy * kwargs['crop_margin'] + 1
    if x0 < 0:
        x0 = 0
    if x1 > size_x:
        x1 = size_x
    if y0 < 0:
        y0 = 0
    if y1 > size_y:
        y1 = size_y
    dx = x1 - x0
    dy = y1 - y0
    if dx > dy * kwargs['anisotropy']:
        dy = 0.5 * (dx / kwargs['anisotropy'] - dy)
        y0 -= dy
        y1 += dy
    else:
        dx = 0.5 * (dy * kwargs['anisotropy'] - dx)
        x0 -= dx
        x1 += dx

    # Generate the transformation matrix
    trans = np.array([[1, 0, -0.5 * img_shape[0]], [0, 1, -0.5 * img_shape[1]],
                      [0, 0, 1]])
    trans = np.dot(
        np.array([[(y1 - y0) / img_shape[0], 0, 0],
                  [0, (x1 - x0) / img_shape[1], 0], [0, 0, 1]]), trans)
    if augment:
        trans = np.dot(
            build_transform(
                random.uniform(-5, 5), random.uniform(-5, 5),
                random.uniform(0.8, 1.0), random.uniform(0.8, 1.0),
                random.uniform(-0.05 * (y1 - y0), 0.05 * (y1 - y0)),
                random.uniform(-0.05 * (x1 - x0), 0.05 * (x1 - x0))), trans)
    trans = np.dot(
        np.array([[1, 0, 0.5 * (y1 + y0)], [0, 1, 0.5 * (x1 + x0)],
                  [0, 0, 1]]), trans)

    # Read the image, transform to black and white and comvert to numpy array
    img = read_raw_image(p).convert('L')
    img = keras.preprocessing.image.img_to_array(img)

    # Apply affine transformation
    matrix = trans[:2, :2]
    offset = trans[:2, 2]
    img = img.reshape(img.shape[:-1])
    img = affine_transform(
        img,
        matrix,
        offset,
        output_shape=img_shape[:-1],
        order=1,
        mode='constant',
        cval=np.average(img))
    img = img.reshape(img_shape)

    # Normalize to zero mean and unit variance
    img -= np.mean(img, keepdims=True)
    img /= np.std(img, keepdims=True) + keras.backend.epsilon()
    return img


if __name__ == "__main__":
    tagged = dict([(p, w) for _, p, w in read_csv(TRAIN_DF).to_records()])
    # submit = [p for _, p, _ in read_csv(SUB_Df).to_records()]
    # join = list(tagged.keys()) + submit
    # p2h, p2size, h2p, w2hs, w2ts, h2ws, t2i, train = load_meta()
    # print(len(h2p), list(h2p.items())[:5])
    # for p in list(tagged.keys()):
    #     h = p2h[p]
    pass