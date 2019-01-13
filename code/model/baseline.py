from tensorflow import keras


def subblock(x, nb_filter, **kwargs):
    x = keras.layers.BatchNormalization()(x)
    y = x
    # Reduce the number of features to 'nb_filter'
    y = keras.layers.Conv2D(nb_filter, (1, 1), activation='relu', **kwargs)(y)
    y = keras.layers.BatchNormalization()(y)

    # Extend the feature field
    y = keras.layers.Conv2D(nb_filter, (3, 3), activation='relu', **kwargs)(y)
    y = keras.layers.BatchNormalization()(y)

    # no activation. Restore the number of original features
    y = keras.layers.Conv2D(keras.backend.int_shape(x)[-1], (1, 1), **kwargs)(y)
    y = keras.layers.Add()([x, y])  # Add the bypass connection
    y = keras.layers.Activation('relu')(y)
    return y


def build_model(img_shape, lr, l2, activation='sigmoid'):
    ##############
    # BRANCH MODEL
    ##############
    regul = keras.regularizers.l2(l2)
    optim = keras.optimizers.Adam(lr=lr)
    kwargs = {'padding': 'same', 'kernel_regularizer': regul}

    inp = keras.layers.Input(shape=img_shape)  # 384x384x1
    x = keras.layers.Conv2D(64, (9, 9), strides=2, activation='relu', **kwargs)(inp)

    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(x)  # 96x96x64
    for _ in range(2):
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Conv2D(64, (3, 3), activation='relu', **kwargs)(x)

    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(x)  # 48x48x64
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(128, (1, 1), activation='relu', **kwargs)(x)  # 48x48x128
    for _ in range(4):
        x = subblock(x, 64, **kwargs)

    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(x)  # 24x24x128
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(256, (1, 1), activation='relu', **kwargs)(x)  # 24x24x256
    for _ in range(4):
        x = subblock(x, 64, **kwargs)

    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(x)  # 12x12x256
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(384, (1, 1), activation='relu', **kwargs)(x)  # 12x12x384
    for _ in range(4):
        x = subblock(x, 96, **kwargs)

    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(x)  # 6x6x384
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(512, (1, 1), activation='relu', **kwargs)(x)  # 6x6x512
    for _ in range(4):
        x = subblock(x, 128, **kwargs)

    x = keras.layers.GlobalMaxPooling2D()(x)  # 512
    branch_model = keras.Model(inp, x)

    ############
    # HEAD MODEL
    ############
    mid = 32
    xa_inp = keras.layers.Input(shape=branch_model.output_shape[1:])
    xb_inp = keras.layers.Input(shape=branch_model.output_shape[1:])
    x1 = keras.layers.Lambda(lambda x: x[0] * x[1])([xa_inp, xb_inp])
    x2 = keras.layers.Lambda(lambda x: x[0] + x[1])([xa_inp, xb_inp])
    x3 = keras.layers.Lambda(lambda x: keras.backend.abs(x[0] - x[1]))([xa_inp, xb_inp])
    x4 = keras.layers.Lambda(lambda x: keras.backend.square(x))(x3)
    x = keras.layers.Concatenate()([x1, x2, x3, x4])
    x = keras.layers.Reshape((4, branch_model.output_shape[1], 1), name='reshape1')(x)

    # Per feature NN with shared weight is implemented using CONV2D with appropriate stride.
    x = keras.layers.Conv2D(mid, (4, 1), activation='relu', padding='valid')(x)
    x = keras.layers.Reshape((branch_model.output_shape[1], mid, 1))(x)
    x = keras.layers.Conv2D(1, (1, mid), activation='linear', padding='valid')(x)
    x = keras.layers.Flatten(name='flatten')(x)

    # Weighted sum implemented as a Dense layer.
    x = keras.layers.Dense(
        1, use_bias=True, activation=activation, name='weighted-average')(x)
    head_model = keras.Model([xa_inp, xb_inp], x, name='head')

    ########################
    # SIAMESE NEURAL NETWORK
    ########################
    # Complete model is constructed by calling the branch model on each input image,
    # and then the head model on the resulting 512-vectors.
    img_a = keras.layers.Input(shape=img_shape)
    img_b = keras.layers.Input(shape=img_shape)
    xa = branch_model(img_a)
    xb = branch_model(img_b)
    x = head_model([xa, xb])
    model = keras.Model([img_a, img_b], x)
    model.compile(
        optim,
        loss='binary_crossentropy',
        metrics=['binary_crossentropy', 'acc'])
    return model, branch_model, head_model


model, branch_model, head_model = build_model((384,384,1), 64e-5, 0)