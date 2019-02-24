import tensorflow as tf
from util import ArcfaceLoss, identity_loss, inference_loss
from tensorflow import keras
import numpy as np
import wn

batch_size = 16
num_classes = 10
epochs = 50

# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Maintain single value ground truth labels for center loss inputs
# Because Embedding layer only accept index as inputs instead of one-hot vector
y_train_value = y_train
y_test_value = y_test

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

inputs = keras.layers.Input(shape=(28, 28, 1))
labels = keras.layers.Input(shape=(1, ), dtype='int32')

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
x = keras.layers.Dense(2, name="embed")(x)
embedding = keras.layers.PReLU(name='ip1')(x)

centers = keras.layers.Embedding(num_classes, 2)(labels)
l2_loss = keras.layers.Lambda(
    lambda x: keras.backend.sum(
        keras.backend.square(x[0] - x[1][:, 0]), 1, keepdims=True),
    name='l2_loss')([embedding, centers])

out = keras.layers.Dense(
    num_classes, activation="softmax", name="prediction")(embedding)

model = keras.Model(inputs=[inputs, labels], outputs=[out, l2_loss])

opt = keras.optimizers.Adam(lr=0.001)

tensorboard = keras.callbacks.TensorBoard(
    log_dir="./testlogs",
    batch_size=batch_size,
    embeddings_freq=1,
    embeddings_layer_names=['ip1'],
    embeddings_data=[x_test, y_test_value])
model.compile(
    loss=["categorical_crossentropy", identity_loss],
    optimizer=opt,
    metrics=dict(prediction="accuracy"),
    loss_weights=[1, .2])
# wn.data_based_init(model, x_train[:100])
model.fit([x_train, y_train_value],
          [y_train, np.zeros(y_train_value.shape)],
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=([x_test, y_test_value],
                           [y_test, np.zeros(y_test_value.shape)]),
          callbacks=[
              keras.callbacks.ModelCheckpoint(
                  "./testlogs",
                  monitor="val_loss",
                  verbose=1,
                  save_best_only=True,
                  save_weights_only=True,
                  mode="min"),
              keras.callbacks.TensorBoard(
                  log_dir="./testlogs",
                  batch_size=batch_size,
                  embeddings_freq=1,
                  embeddings_layer_names=['embed'],
                  embeddings_data=[x_test, y_test_value])
          ])
# # save class labels to disk to color data points in TensorBoard accordingly
# with open(join(log_dir, 'metadata.tsv'), 'w') as f:
#     np.savetxt(f, y_test)
