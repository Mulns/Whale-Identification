from models import CenterLossNet
from data import ImageDataLabelGenerator, rgb2ycbcr
from backbones import siamise

H, W, C = (150, 300, 3)
nb_classes = 5004
lambda_c = 0.2
feature_size = 512

from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


def pca(n_neighbors=6, is_pca=True):
    #get train and test  x:data;y:label
    y_train = np.arange(0, nb_classes)
    x_train = model.get_centers().predict(y_train).squeeze()
    test_data = next(valid_gen)
    test_embed = model.get_embedding().predict(test_data[0])
    x_test = test_embed
    y_test = test_data[0][1]

    # print("x_train: ", x_train.shape)
    # print("y_train: ", y_train.shape)
    # print("x_test: ", x_test.shape)
    # print("y_test: ", y_test.shape)
    #train PCA model
    if is_pca:
        pca = PCA(n_components=100).fit(x_train)

        # return data after pca
        x_train_pca = pca.transform(x_train)
        x_test_pca = pca.transform(x_test)
    else:
        x_train_pca = x_train
        x_test_pca = x_test

    #knn core
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)

    #train the model using train dataset
    knn.fit(x_train_pca, y_train)

    #test the data
    y_test_predict = knn.predict(x_test_pca)

    #predict the accuracy rate
    print("%d, %s, score of knn: " % (n_neighbors, str(is_pca)),
          knn.score(x_test_pca, y_test))


def softmax():
    test_data = next(valid_gen)
    nbs = model.model.evaluate(test_data[0], test_data[1])
    print("acc of softmax: ", nbs[-1])


if __name__ == "__main__":
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
        batch_size=399,
        subset="validation",
        shuffle=True,
        interpolation="bicubic")
    model = CenterLossNet(siamise, "./trainSpace/",
                          "CenterLossNet").create_model(
                              _compile=True,
                              use_weightnorm=False,
                              database_init=False,
                              load_weights=True,
                              lambda_c=lambda_c)
    for k in range(1, 10):
        for is_pca in [True, False]:
            pca(n_neighbors=k, is_pca=is_pca)
    softmax()