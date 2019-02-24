from tensorflow import keras
from tensorflow.python.keras import callbacks
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import layers
from tensorflow.python.keras import backend as K
from tensorflow import keras
weight_decay = 5e-4


def Vgg16(img_input):
    x = keras.layers.Conv2D(
        64, (3, 3),
        activation='relu',
        padding='same',
        kernel_regularizer=keras.regularizers.l2(l=weight_decay),
        name='block1_conv1')(img_input)
    x = keras.layers.Conv2D(
        64, (3, 3),
        activation='relu',
        padding='same',
        kernel_regularizer=keras.regularizers.l2(l=weight_decay),
        name='block1_conv2')(x)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2),
                                  name='block1_pool')(x)

    # Block 2
    x = keras.layers.Conv2D(
        128, (3, 3),
        activation='relu',
        padding='same',
        kernel_regularizer=keras.regularizers.l2(l=weight_decay),
        name='block2_conv1')(x)
    x = keras.layers.Conv2D(
        128, (3, 3),
        activation='relu',
        padding='same',
        kernel_regularizer=keras.regularizers.l2(l=weight_decay),
        name='block2_conv2')(x)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2),
                                  name='block2_pool')(x)

    # Block 3
    x = keras.layers.Conv2D(
        256, (3, 3),
        activation='relu',
        padding='same',
        kernel_regularizer=keras.regularizers.l2(l=weight_decay),
        name='block3_conv1')(x)
    x = keras.layers.Conv2D(
        256, (3, 3),
        activation='relu',
        padding='same',
        kernel_regularizer=keras.regularizers.l2(l=weight_decay),
        name='block3_conv2')(x)
    x = keras.layers.Conv2D(
        256, (3, 3),
        activation='relu',
        padding='same',
        kernel_regularizer=keras.regularizers.l2(l=weight_decay),
        name='block3_conv3')(x)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2),
                                  name='block3_pool')(x)

    # Block 4
    x = keras.layers.Conv2D(
        512, (3, 3),
        activation='relu',
        padding='same',
        kernel_regularizer=keras.regularizers.l2(l=weight_decay),
        name='block4_conv1')(x)
    x = keras.layers.Conv2D(
        512, (3, 3),
        activation='relu',
        padding='same',
        kernel_regularizer=keras.regularizers.l2(l=weight_decay),
        name='block4_conv2')(x)
    x = keras.layers.Conv2D(
        512, (3, 3),
        activation='relu',
        padding='same',
        kernel_regularizer=keras.regularizers.l2(l=weight_decay),
        name='block4_conv3')(x)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2),
                                  name='block4_pool')(x)

    # Block 5
    x = keras.layers.Conv2D(
        512, (3, 3),
        activation='relu',
        padding='same',
        kernel_regularizer=keras.regularizers.l2(l=weight_decay),
        name='block5_conv1')(x)
    x = keras.layers.Conv2D(
        512, (3, 3),
        activation='relu',
        padding='same',
        kernel_regularizer=keras.regularizers.l2(l=weight_decay),
        name='block5_conv2')(x)
    x = keras.layers.Conv2D(
        512, (3, 3),
        activation='relu',
        padding='same',
        kernel_regularizer=keras.regularizers.l2(l=weight_decay),
        name='block5_conv3')(x)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2),
                                  name='block5_pool')(x)

    x = keras.layers.Flatten(name='flatten')(x)
    x = keras.layers.Dense(
        4096,
        activation='relu',
        name='fc1',
        kernel_regularizer=keras.regularizers.l2(l=weight_decay))(x)
    x = keras.layers.Dense(
        160,
        activation='relu',
        name='embedding_out',
        kernel_regularizer=keras.regularizers.l2(l=weight_decay))(x)

    return x


def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(
        filters1, (1, 1),
        kernel_initializer='he_normal',
        name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(
        filters2,
        kernel_size,
        padding='same',
        kernel_initializer='he_normal',
        name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(
        filters3, (1, 1),
        kernel_initializer='he_normal',
        name=conv_name_base + '2c')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)
    return x


def conv_block(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               strides=(2, 2)):
    """A block that has a conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the first conv layer in the block.

    # Returns
        Output tensor for the block.

    Note that from stage 3,
    the first conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(
        filters1, (1, 1),
        strides=strides,
        kernel_initializer='he_normal',
        name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(
        filters2,
        kernel_size,
        padding='same',
        kernel_initializer='he_normal',
        name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(
        filters3, (1, 1),
        kernel_initializer='he_normal',
        name=conv_name_base + '2c')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = layers.Conv2D(
        filters3, (1, 1),
        strides=strides,
        kernel_initializer='he_normal',
        name=conv_name_base + '1')(input_tensor)
    shortcut = layers.BatchNormalization(
        axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return x


def resnet50(img_input):
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
    x = layers.Conv2D(
        64, (7, 7),
        strides=(2, 2),
        padding='valid',
        kernel_initializer='he_normal',
        name='conv1')(x)
    x = layers.BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = layers.Activation('relu')(x)
    x = layers.ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
    x = layers.Flatten(name='flatten')(x)
    x = layers.Dense(160, activation="relu", name="embedding_out")(x)
    return x


def simple(inputs):

    x = keras.layers.Conv2D(32, (3, 3))(inputs)
    x = keras.layers.PReLU()(x)
    x = keras.layers.Conv2D(32, (3, 3))(x)
    x = keras.layers.PReLU()(x)
    x = keras.layers.Conv2D(64, (3, 3))(x)
    x = keras.layers.PReLU()(x)
    x = keras.layers.Conv2D(64, (5, 5))(x)
    x = keras.layers.PReLU()(x)
    x = keras.layers.Conv2D(128, (5, 5))(x)
    x = keras.layers.PReLU()(x)
    x = keras.layers.Conv2D(128, (5, 5))(x)
    x = keras.layers.PReLU()(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(160, name="embed")(x)
    embedding = keras.layers.PReLU(name='ip1')(x)
    return embedding


def subblock(x, filter, **kwargs):
    # x = keras.layers.BatchNormalization()(x)
    y = x
    y = keras.layers.Conv2D(
        filter, (1, 1), activation='selu',
        **kwargs)(y)  # Reduce the number of features to 'filter'
    # y = keras.layers.BatchNormalization()(y)
    y = keras.layers.Conv2D(
        filter, (3, 3), activation='selu',
        **kwargs)(y)  # Extend the feature field
    # y = keras.layers.BatchNormalization()(y)
    y = keras.layers.Conv2D(K.int_shape(x)[-1], (1, 1), **kwargs)(
        y)  # no activation # Restore the number of original features
    y = keras.layers.Add()([x, y])  # Add the bypass connection
    y = keras.layers.Activation('selu')(y)
    return y


def siamise(inp):
    kwargs = {
        'padding': 'same',
        'kernel_regularizer': keras.regularizers.l2(weight_decay)
    }
    # 384x384x1
    x = keras.layers.Conv2D(
        64, (9, 9), strides=2, activation='selu', **kwargs)(inp)

    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(x)  # 96x96x64
    for _ in range(2):
        # x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Conv2D(64, (3, 3), activation='selu', **kwargs)(x)

    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(x)  # 48x48x64
    # x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(
        128, (1, 1), activation='selu', **kwargs)(x)  # 48x48x128
    for _ in range(4):
        x = subblock(x, 64, **kwargs)

    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(x)  # 24x24x128
    # x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(
        256, (1, 1), activation='selu', **kwargs)(x)  # 24x24x256
    for _ in range(4):
        x = subblock(x, 64, **kwargs)

    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(x)  # 12x12x256
    # x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(
        384, (1, 1), activation='selu', **kwargs)(x)  # 12x12x384
    for _ in range(4):
        x = subblock(x, 96, **kwargs)

    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(x)  # 6x6x384
    # x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(
        512, (1, 1), activation='selu', **kwargs)(x)  # 6x6x512
    for _ in range(4):
        x = subblock(x, 128, **kwargs)

    x = keras.layers.GlobalMaxPooling2D()(x)  # 512

    return x