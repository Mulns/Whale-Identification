import tensorflow as tf
from tensorflow import keras
import numpy as np
from contextlib import contextmanager
from PIL import Image
from data import FeatureGen, ScoreGen



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


def score_reshape(score, x, y=None):
    """
    Tranformed the packed matrix 'score' into a square matrix.
    @param score the packed matrix
    @param x the first image feature tensor
    @param y the second image feature tensor if different from x
    @result the square matrix
    """
    if y is None:
        # When y is None, score is a packed upper triangular matrix.
        # Unpack, and transpose to form the symmetrical lower triangular matrix.
        m = np.zeros((x.shape[0], x.shape[0]), dtype=keras.backend.floatx())
        m[np.triu_indices(x.shape[0], 1)] = score.squeeze()
        m += m.transpose()
    else:
        m = np.zeros((y.shape[0], x.shape[0]), dtype=keras.backend.floatx())
        iy, ix = np.indices((y.shape[0], x.shape[0]))
        ix = ix.reshape((ix.size, ))
        iy = iy.reshape((iy.size, ))
        m[iy, ix] = score.squeeze()
    return m

def compute_score(data, shape, branch_model, head_model, verbose=1, **metadata):
    """
    Compute the score matrix by scoring every pictures from the training set against every other picture O(n^2).
    """
    features = branch_model.predict_generator(
        FeatureGen(data, shape, verbose=verbose, **metadata),
        max_queue_size=12,
        workers=6,
        verbose=0)
    score = head_model.predict_generator(
        ScoreGen(features, verbose=verbose),
        max_queue_size=12,
        workers=6,
        verbose=0)
    score = score_reshape(score, features)
    return features, score