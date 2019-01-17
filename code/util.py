import tensorflow as tf
from tensorflow import keras

from contextlib import contextmanager
from PIL import Image


@contextmanager
def concurrent_generator(sequence,
                         num_workers=8,
                         max_queue_size=32,
                         use_multiprocessing=False):
    enqueuer = keras.utils.OrderedEnqueuer(
        sequence, use_multiprocessing=use_multiprocessing)
    try:
        enqueuer.start(workers=num_workers, max_queue_size=max_queue_size)
        yield enqueuer.get()
    finally:
        enqueuer.stop()


def init_session(gpu_memory_fraction):
    keras.backend.set_session(
        _tensorflow_session(gpu_memory_fraction=gpu_memory_fraction))


def reset_session(gpu_memory_fraction):
    keras.backend.clear_session()
    init_session(gpu_memory_fraction)


def _tensorflow_session(gpu_memory_fraction):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_fraction
    return tf.Session(config=config)


def mae(hr, sr):
    hr, sr = _crop_hr_in_training(hr, sr)
    return keras.losses.mean_absolute_error(hr, sr)


def psnr(hr, sr):
    hr, sr = _crop_hr_in_training(hr, sr)
    return tf.image.psnr(hr, sr, max_val=255)


def _crop_hr_in_training(hr, sr):
    """
    Remove margin of size scale*2 from hr in training phase.

    The margin is computed from size difference of hr and sr
    so that no explicit scale parameter is needed. This is only
    needed for WDSR models.
    """

    margin = (tf.shape(hr)[1] - tf.shape(sr)[1]) // 2

    # crop only if margin > 0
    hr_crop = tf.cond(
        tf.equal(margin, 0), lambda: hr,
        lambda: hr[:, margin:-margin, margin:-margin, :])

    hr = keras.backend.in_train_phase(hr_crop, hr)
    hr.uses_learning_phase = True
    return hr, sr
