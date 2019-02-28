from tensorflow import keras
from util import ArcfaceLoss, inference_loss, identity_loss
import numpy as np
import os
import wn

weight_decay = 5e-4
H, W, C = (150, 300, 3)
nb_classes = 5004
lambda_c = 0.2
lr = 6e-4
feature_size = 512
final_active = 'sigmoid'  # for siamese net

# labels: not one-hot


class BaseModel(object):
    def __init__(self, workspace, model_name):
        if not os.path.isdir(workspace):
            os.mkdir(workspace)
        self.weights_path = os.path.join(workspace, "weights",
                                         "%s.h5" % model_name)
        self.log_dir = os.path.join(workspace, "logs", model_name)

    def create_model(self, ):
        self.model = None

    def fit(self, train_gen, valid_gen, batch_size, nb_epochs):
        callback_list = [
            keras.callbacks.ModelCheckpoint(
                self.weights_path,
                monitor="val_loss",
                save_best_only=True,
                mode='min',
                save_weights_only=True,
                verbose=2),
            # keras.callbacks.ReduceLROnPlateau(
            #     monitor="val_loss",
            #     factor=self.lr_reduce_factor,
            #     patience=self.lr_reduce_patience,
            #     verbose=1,
            #     mode='min',
            #     epsilon=1e-6,
            #     cooldown=0,
            #     min_lr=1e-6),
            keras.callbacks.TensorBoard(
                log_dir=self.log_dir,
                histogram_freq=10,
                write_grads=False,
                write_graph=False,
                write_images=False,
                batch_size=10)
        ]
        self.model.fit_generator(
            generator=train_gen,
            steps_per_epoch=2000,
            epochs=nb_epochs,
            verbose=1,
            callbacks=callback_list,
            validation_data=valid_gen,
            validation_steps=20,
            max_queue_size=6,
            workers=6,
            use_multiprocessing=True,
            shuffle=True)


class ArcFace(BaseModel):
    def __init__(self, backbone, workspace, model_name):
        super(ArcFace, self).__init__(workspace, model_name)
        self.backbone = backbone
        self.weight_norm = False

    def create_model(self,
                     _compile=True,
                     load_weights=False,
                     weights_path=None):
        images = keras.layers.Input(shape=(H, W, C))
        labels_ = keras.layers.Input(shape=(1, ), dtype='int32')
        labels = keras.layers.Lambda(
            lambda x: keras.backend.squeeze(x, axis=1),
            name="squeeze")(labels_)

        embedding = self.backbone(images)
        logit = ArcfaceLoss(
            nb_classes, m=0.5, s=64., name="arcface_loss")([embedding, labels])
        # print(keras.backend.int_shape(logit))
        pred = keras.layers.Softmax(name="prediction")(logit)

        # print(keras.backend.int_shape(pred))
        loss = keras.layers.Lambda(
            inference_loss, name="softmax_loss")([logit, labels])

        self.model = keras.Model(
            inputs=[images, labels_], outputs=[pred, loss])

        if _compile:
            # compilation
            # if self.weight_norm:
            #     opt = wn.AdamWithWeightnorm(lr=self.learning_rate)
            # else:
            #     opt = keras.optimizers.Adam(
            #         lr=self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
            opt = keras.optimizers.SGD(lr=1e-4, momentum=0.9, decay=0.)
            self.model.compile(
                optimizer=opt,
                loss=["categorical_crossentropy", identity_loss],
                metrics=dict(prediction="accuracy"),
                loss_weights=[0, 1])
            # if self.weight_norm and self.num_init_batches:
            #     model_weightnorm_init(self.model, tr_gen, self.num_init_batches)
        if load_weights:
            weights_path = self.weights_path if weights_path is None else weights_path
            self.model.load_weights(weights_path, by_name=True)

        return self


class CenterLossNet(BaseModel):
    def __init__(self, backbone, workspace, model_name):
        super(CenterLossNet, self).__init__(workspace, model_name)
        self.backbone = backbone

    def create_model(self,
                     _compile=True,
                     use_weightnorm=False,
                     database_init=False,
                     data=None,
                     lambda_c=0.2,
                     load_weights=False,
                     weights_path=None):
        images = keras.layers.Input(shape=(H, W, C))
        labels = keras.layers.Input(shape=(1, ), dtype='int32')
        # labels = keras.layers.Lambda(
        #     lambda x: keras.backend.squeeze(x, axis=1), name="squeeze")(labels_)

        centers = keras.layers.Embedding(nb_classes, feature_size)(labels)
        embedding = self.backbone(images)
        l2_loss = keras.layers.Lambda(
            lambda x: keras.backend.sum(
                keras.backend.square(x[0] - x[1][:, 0]), 1, keepdims=True),
            name='l2_loss')([embedding, centers])

        out = keras.layers.Dense(
            nb_classes, activation="softmax", name="prediction")(embedding)

        self.model = keras.Model(
            inputs=[images, labels], outputs=[out, l2_loss])

        if _compile:
            # compilation
            if use_weightnorm:
                opt = wn.AdamWithWeightnorm(lr=lr)
            else:
                opt = keras.optimizers.Adam(
                    lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
            # opt = keras.optimizers.SGD(lr=1e-4, momentum=0.9, decay=0.)
            self.model.compile(
                optimizer=opt,
                loss=["categorical_crossentropy", identity_loss],
                metrics=dict(prediction="accuracy"),
                loss_weights=[1, lambda_c])
            if use_weightnorm and database_init:
                wn.data_based_init(self.model, data)
        if load_weights:
            weights_path = self.weights_path if weights_path is None else weights_path
            self.model.load_weights(weights_path, by_name=True)

        return self

    def get_embedding(self, ):
        return keras.Model(self.model.get_layer("input_1").output,
                           self.model.get_layer("prediction").input)

    def get_centers(self, ):
        # (N,) integar numpy array
        return keras.Model(
            self.model.get_layer("input_2").input,
            self.model.get_layer("embedding").output)


class Siamese(BaseModel):
    def __init__(self, backbone, workspace, model_name):
        super(Siamese, self).__init__(workspace, model_name)
        self.backbone = backbone

    def create_branch(self, ):
        init = keras.layers.Input((H, W, C))
        out = self.backbone(init)
        return keras.Model(init, out)

    def create_model(self,
                     _compile=True,
                     use_weightnorm=False,
                     database_init=False,
                     data=None,
                     lambda_c=0.2,
                     load_weights=False,
                     weights_path=None):
        i1 = keras.layers.Input((H, W, C))
        i2 = keras.layers.Input((H, W, C))
        branch = self.create_branch()
        branch.trainable = False
        xa_inp = branch(i1)
        xb_inp = branch(i2)

        x1 = keras.layers.Lambda(lambda x: x[0] * x[1])([xa_inp, xb_inp])
        x2 = keras.layers.Lambda(lambda x: x[0] + x[1])([xa_inp, xb_inp])
        x3 = keras.layers.Lambda(lambda x: keras.backend.abs(x[0] - x[1]))(
            [xa_inp, xb_inp])
        x4 = keras.layers.Lambda(lambda x: keras.backend.square(x))(x3)
        x = keras.layers.Concatenate()([x1, x2, x3, x4])
        x = keras.layers.Reshape((4, feature_size, 1), name='reshape1')(x)

        # Per feature NN with shared weight is implemented using CONV2D with appropriate stride.
        x = keras.layers.Conv2D(
            32, (4, 1), activation='relu', padding='valid')(x)
        x = keras.layers.Reshape((feature_size, 32, 1))(x)
        x = keras.layers.Conv2D(
            1, (1, 32), activation='linear', padding='valid')(x)
        x = keras.layers.Flatten(name='flatten')(x)

        # Weighted sum implemented as a Dense layer.
        score = keras.layers.Dense(
            1, use_bias=True, activation=final_active,
            name='weighted-average')(x)

        self.model = keras.Model([i1, i2], score)
        if _compile:
            # compilation
            if use_weightnorm:
                opt = wn.AdamWithWeightnorm(lr=lr)
            else:
                opt = keras.optimizers.Adam(
                    lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
            # opt = keras.optimizers.SGD(lr=1e-4, momentum=0.9, decay=0.)
            self.model.compile(
                optimizer=opt,
                loss=["binary_crossentropy"],
                metrics=dict(prediction="weighted-average"),
                loss_weights=[1])
            if use_weightnorm and database_init:
                wn.data_based_init(self.model, data)
        if load_weights:
            weights_path = self.weights_path if weights_path is None else weights_path
            self.model.load_weights(weights_path, by_name=True)

        return self


if __name__ == "__main__":
    from backbones import Vgg16, resnet50, siamese
    from data import rgb2ycbcr, ImageDataLabelGenerator

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
        batch_size=10,
        subset="validation",
        shuffle=True,
        interpolation="bicubic")

    model = Siamese(siamese, "./trainSpace/", "SiameseNet").create_model(
        _compile=True,
        use_weightnorm=False,
        database_init=False,
        load_weights=True,
        weights_path="./trainSpace/weights/CenterLossNet.h5",
        lambda_c=lambda_c)
    # model.fit(train_gen, valid_gen, batch_size=10, nb_epochs=10000)
