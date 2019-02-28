from backbones import Vgg16, resnet50, siamese
from data import rgb2ycbcr, ImageDataLabelGenerator
from models import CenterLossNet
from tensorflow import keras
import numpy as np
from PIL import Image
import pickle
import os

weight_decay = 5e-4
H, W, C = (150, 300, 3)
nb_classes = 5004
lambda_c = 0.2
lr = 6e-4
feature_size = 512
final_active = 'sigmoid'  # for siamese net

train_data_gen = keras.preprocessing.image.ImageDataGenerator(
    samplewise_center=True,
    samplewise_std_normalization=True,
    zca_whitening=False,
    zca_epsilon=1e-6,
    rotation_range=16,
    width_shift_range=0.2,
    height_shift_range=0.1,
    zoom_range=0.2,
    fill_mode='reflect',
    horizontal_flip=True,
    vertical_flip=False,
    preprocessing_function=rgb2ycbcr,
    rescale=1. / 255,
    validation_split=0.1)

model = CenterLossNet(siamese, "./trainSpace/", "CenterLossNet").create_model(
    _compile=True,
    use_weightnorm=False,
    database_init=False,
    load_weights=True,
    weights_path="./trainSpace/weights/CenterLossNet.h5",
    lambda_c=lambda_c).get_embedding()

with open("../Dataset/metadata/p2l.pickle", "rb") as f:
  p2l = pickle.load(f)
with open("../Dataset/metadata/tr_l2ps.pickle", "rb") as f:
  l2ps = pickle.load(f)

def cal_dis(pa, pb):
    batch_x = []
    for p in [pa, pb]:
        img = keras.preprocessing.image.load_img(
            os.path.join("../Dataset/train", p),
            color_mode='rgb',
            target_size=(H,W),
            interpolation="bicubic")
        x = keras.preprocessing.image.img_to_array(
            img, data_format="channels_last")
        if hasattr(img, 'close'):
            img.close()
        x = train_data_gen.standardize(x)
        batch_x.append(x)
    batch_embed = model.predict(np.array(batch_x))
    dis = np.sqrt(np.sum(np.square(batch_embed[0]-batch_embed[1])))
    return dis


match = {}
for p, l in p2l.items():
    if l in l2ps.keys():
        match[p] = l2ps[l]
unmatch = {}
for p, l in p2l.items():
    dis_p = []
    for p1, l1 in p2l.items():
        if l1 != l:
            dis_p.append((p1, cal_dis(p, p1)))
    dis_p = sorted(dis_p, key=lambda x:x[1], reverse=True)
    dis_p = dis_p[:10]
    ps1, _ = zip(*dis_p)
    unmatch[p] = ps1

