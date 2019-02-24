import numpy as np
from scipy.ndimage import affine_transform
from PIL import Image
import random
import pickle
from tensorflow import keras
import os
import math
from matplotlib import pyplot as plt
data_dir = "../Dataset/train"
nb_classes = 5004


def rgb2ycbcr(image):
    """Transfer RGB image to YCbCr.
    
        Args:
            image: Numpy array in range of (0, 255)
    
        Returns:
            final_img: Numpy array in float64.
    
        Raises:
            ValueError: An error occured when input image is not RGB mode.
            ValueError: An error occured when input image is not in range of (0, 255).
    """
    assert image.shape[-1] == 3, "Input should be in RGB mode."
    # assert np.max(image) > 1., "Input should be in range of (0, 255)"
    R, G, B = [image[:, :, i][:, :, np.newaxis] for i in range(3)]
    Y = 0.257 * R + 0.504 * G + 0.098 * B + 16
    Cb = -0.148 * R - 0.291 * G + 0.439 * B + 128
    Cr = 0.439 * R - 0.368 * G - 0.071 * B + 128
    final_img = np.concatenate((Y, Cb, Cr), axis=2)
    # return np.uint8(final_img)
    return final_img.squeeze()


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


class WhaleSequence(keras.utils.Sequence):
    def __init__(self,
                 p2l,
                 p2bb,
                 data_dir,
                 img_shape,
                 batch_size,
                 shuffle=True,
                 augment=True):

        self.p2l = p2l
        self.ps = [p for p in p2l.keys()]
        self.nb_data = len(self.ps)
        if shuffle:
            random.shuffle(self.ps)

        self.img_shape = img_shape
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.augment = augment
        self.p2bb = p2bb

    def preprocess(self, p, crop_margin=0.05, anisotropy=2.55):
        img = np.array(Image.open(os.path.join(self.data_dir, p)))
        size_y, size_x = img.shape[:2]

        row = self.p2bb.loc[p]
        x0, y0, x1, y1 = row['x0'], row['y0'], row['x1'], row['y1']
        # dx = x1 - x0
        # dy = y1 - y0
        # x0 -= dx * 0.05
        # y0 -= dy * 0.05
        # y1 += dy * 0.05 + 1
        # x1 += dx * 0.05 + 1
        # if x0 < 0:
        #     x0 = 0
        # if x1 > size_x:
        #     x1 = size_x
        # if y0 < 0:
        #     y0 = 0
        # if y1 > size_y:
        #     y1 = size_y
        # dx = x1 - x0
        # dy = y1 - y0
        # if dx > dy * anisotropy:
        #     dy = 0.5 * (dx / anisotropy - dy)
        #     y0 -= dy
        #     y1 += dy
        # else:
        #     dx = 0.5 * (dy * anisotropy - dx)
        #     x0 -= dx
        #     x1 += dx
        # Generate the transformation matrix
        # # XXX Why???
        # trans = np.array([[1, 0, -0.5 * self.img_shape[0]],
        #                   [0, 1, -0.5 * self.img_shape[1]], [0, 0, 1]])
        # trans = np.dot(
        #     np.array([[(y1 - y0) / self.img_shape[0], 0, 0],
        #               [0, (x1 - x0) / self.img_shape[1], 0], [0, 0, 1]]),
        #     trans)
        # if self.augment:
        #     trans = np.dot(
        #         build_transform(
        #             random.uniform(-5, 5), random.uniform(-5, 5),
        #             random.uniform(0.8, 1.0), random.uniform(0.8, 1.0),
        #             random.uniform(-0.05 * (y1 - y0), 0.05 * (y1 - y0)),
        #             random.uniform(-0.05 * (x1 - x0), 0.05 * (x1 - x0))),
        #         trans)
        # trans = np.dot(
        #     np.array([[1, 0, 0.5 * (y1 + y0)], [0, 1, 0.5 * (x1 + x0)],
        #               [0, 0, 1]]), trans)

        # Read the image, transform to black and white and comvert to numpy array
        if len(img.shape) == 3:
            img = rgb2ycbcr(img)
        else:
            img = np.concatenate([img[:, :, np.newaxis] for _ in range(3)],
                                 axis=-1)
            img = np.array(img, dtype="float32")

        # # Apply affine transformation
        # matrix = trans[:2, :2]
        # offset = trans[:2, 2]
        # img = affine_transform(
        #     img,
        #     matrix,
        #     offset,
        #     output_shape=self.img_shape[:-1],
        #     order=1,
        #     mode='constant',
        #     cval=np.average(img))
        # print([x0, x1, y0, y1])
        img = img[int(y0):int(y1), int(x0):int(x1), :]
        # img = img.reshape(self.img_shape)

        # Normalize to zero mean and unit variance
        img -= np.mean(img, keepdims=True)
        img /= np.std(img, keepdims=True) + keras.backend.epsilon()
        plt.imshow(img.squeeze())
        plt.show()
        return img

    def __len__(self):
        return math.ceil(len(self.ps) / self.batch_size)

    def __getitem__(self, idx):
        batch_p = self.ps[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_data = []
        batch_label = []
        for p in batch_p:
            data = self.preprocess(p, self.img_shape)
            label = self.p2l[p]
            batch_data.append(data)
            batch_label.append(label)
        batch_data = np.array(batch_data)
        batch_label = np.array(batch_label)
        curr_batch_size = batch_label.shape[0]
        return ([batch_data, batch_label], [
            keras.utils.to_categorical(batch_label, nb_classes),
            np.zeros(batch_label.shape)
        ])

    def on_epoch_end(self):
        """Method called at the end of every epoch.
        """
        pass


class MyDirectoryIterator(keras.preprocessing.image.DirectoryIterator):
    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = np.zeros(
            (len(index_array), ) + self.image_shape, dtype=self.dtype)
        # build batch of image data
        for i, j in enumerate(index_array):
            fname = self.filenames[j]
            img = keras.preprocessing.image.load_img(
                os.path.join(self.directory, fname),
                color_mode=self.color_mode,
                target_size=self.target_size,
                interpolation=self.interpolation)
            x = keras.preprocessing.image.img_to_array(
                img, data_format=self.data_format)
            # Pillow images should be closed after `load_img`,
            # but not PIL images.
            if hasattr(img, 'close'):
                img.close()
            params = self.image_data_generator.get_random_transform(x.shape)
            x = self.image_data_generator.apply_transform(x, params)
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x
        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i, j in enumerate(index_array):
                img = keras.preprocessing.image.array_to_img(
                    batch_x[i], self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(
                    prefix=self.save_prefix,
                    index=j,
                    hash=np.random.randint(1e7),
                    format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        # build batch of labels
        if self.class_mode == 'input':
            batch_y = batch_x.copy()
        elif self.class_mode == 'sparse':
            batch_y = self.classes[index_array]
        elif self.class_mode == 'binary':
            batch_y = self.classes[index_array].astype(self.dtype)
        elif self.class_mode == 'categorical':
            batch_y = np.zeros((len(batch_x), self.num_classes),
                               dtype=self.dtype)
            for i, label in enumerate(self.classes[index_array]):
                batch_y[i, label] = 1.
        else:
            return batch_x
        return [batch_x, batch_y], [
            keras.utils.to_categorical(batch_y, nb_classes),
            np.zeros(batch_y.shape)
        ]


class ImageDataLabelGenerator(keras.preprocessing.image.ImageDataGenerator):
    def flow_from_directory(self,
                            directory,
                            target_size=(256, 256),
                            color_mode='rgb',
                            classes=None,
                            class_mode='categorical',
                            batch_size=32,
                            shuffle=True,
                            seed=None,
                            save_to_dir=None,
                            save_prefix='',
                            save_format='png',
                            follow_links=False,
                            subset=None,
                            interpolation='nearest'):
        return MyDirectoryIterator(
            directory,
            self,
            target_size=target_size,
            color_mode=color_mode,
            classes=classes,
            class_mode=class_mode,
            data_format=self.data_format,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            follow_links=follow_links,
            subset=subset,
            interpolation=interpolation)


if __name__ == "__main__":
    import pickle
    from pandas import read_csv
    from data import WhaleSequence
    from backbones import Vgg16
    with open("../Dataset/metadata/p2l_train.pickle", 'rb') as f:
        p2l_train = pickle.load(f)
    with open("../Dataset/metadata/p2l_valid.pickle", 'rb') as f:
        p2l_valid = pickle.load(f)
    p2bb = read_csv("../Dataset/metadata/bounding_boxes.csv").set_index(
        "Image")
    train_data_gen = ImageDataLabelGenerator(
        samplewise_center=False,
        samplewise_std_normalization=False,
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
        target_size=(192, 384),
        color_mode="rgb",
        class_mode="sparse",
        batch_size=16,
        shuffle=True,
        interpolation="bicubic",
        subset='training')
    valid_gen = train_data_gen.flow_from_directory(
        "../Dataset/Train",
        target_size=(192, 384),
        color_mode='rgb',
        class_mode='sparse',
        batch_size=10,
        subset="validation",
        shuffle=True,
        interpolation="bicubic")

    for i, tr in enumerate(train_gen):
        data, label = tr[0]
        label_onehot, losses_v = tr[1]
        
        print(data.shape)
        print(type(label[0]), label.shape)
        print(label_onehot.shape)
        print(losses_v.shape)
        plt.imshow(data[0][:,:,0].squeeze())
        plt.title(label[0])
        plt.show()
        if i > 10:
            break
