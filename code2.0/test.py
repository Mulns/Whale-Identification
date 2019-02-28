from models import CenterLossNet
from backbones import siamise
from data import ImageDataLabelGenerator, rgb2ycbcr
from tensorflow import keras
import numpy as np
from sklearn import decomposition

weight_decay = 5e-4
H, W, C = (150, 300, 3)
nb_classes = 5004
lambda_c = 0.2
lr = 6e-4
feature_size = 512

train_data_gen = ImageDataLabelGenerator(
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
train_gen = train_data_gen.flow_from_directory(
    "../Dataset/Train",
    target_size=(H, W),
    color_mode="rgb",
    class_mode="sparse",
    batch_size=16,
    shuffle=True,
    interpolation="bicubic",
    subset='training')
valid_gen = train_data_gen.flow_from_directory(
    "../Dataset/Train",
    target_size=(H, W),
    color_mode='rgb',
    class_mode='sparse',
    batch_size=1000,
    subset="validation",
    shuffle=True,
    interpolation="bicubic")
model = CenterLossNet(siamise, "./trainSpace/", "CenterLossNet").create_model(
    _compile=True,
    use_weightnorm=False,
    database_init=False,
    load_weights=True,
    lambda_c=lambda_c)
submodel = keras.Model(model.model.inputs,
                       model.model.get_layer("prediction").input)
centers = keras.Model(model.model.get_layer("input_2").input,
                      model.model.get_layer("embedding").output)
for data in iter(valid_gen):
    nb_classes = 100
    embed = submodel.predict(data[0])
    cen = centers.predict(np.arange(0,nb_classes)).squeeze()
    print(embed.shape)
    print(cen.shape)
    pca = decomposition.PCA(n_components=2)
    cen_embed = np.concatenate([cen, embed], axis=0)
    pca.fit(cen_embed)
    cen_embed_pca = pca.transform(cen_embed)
    cen_pca = cen_embed_pca[:nb_classes]
    embed_pca = cen_embed_pca[nb_classes:]
    from matplotlib import pyplot as plt
    plt.scatter(cen_pca[:,0], cen_pca[:,1], marker='o', c='r', alpha=0.5)
    plt.scatter(embed_pca[:,0], embed_pca[:,1], marker='.', c='b')
    plt.show()

    break
keras.utils.plot_model(model.model, "./model.png")
