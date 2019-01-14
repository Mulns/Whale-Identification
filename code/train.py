from tensorflow import keras
from os.path import isfile
from tqdm import tqdm
import numpy as np
import random
from data import FeatureGen, ScoreGen, TrainingData
import image_utils as iu
from .model import baseline

shape = (384, 384, 1)
metadata = iu.load_meta()
train = metadata['train']
print("Training data : ", len(train))
train_set = set(train)
w2hs = metadata['w2hs']
steps = 0
model, branch_model, head_model = baseline.build_model(shape, 64e-5, 0)

# logger = logging.getLogger(__name__)

# def create_train_workspace(path):
#     train_dir = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#     train_dir = os.path.join(path, train_dir)
#     models_dir = os.path.join(train_dir, 'models')
#     os.makedirs(train_dir, exist_ok=True)
#     os.mkdir(models_dir)
#     return train_dir, models_dir

# def write_args(path, args):
#     with open(os.path.join(path, 'args.txt'), 'w') as f:
#         for k, v in sorted(args.__dict__.items()):
#             f.write(f'{k}={v}\n')


def set_lr(model, lr):
    keras.backend.set_value(model.optimizer.lr, float(lr))


def get_lr(model):
    return keras.backend.get_value(model.optimizer.lr)


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


def compute_score(verbose=1):
    """
    Compute the score matrix by scoring every pictures from the training set against every other picture O(n^2).
    """
    features = branch_model.predict_generator(
        FeatureGen(train, shape, verbose=verbose, **metadata),
        max_queue_size=8,
        workers=4,
        verbose=0)
    score = head_model.predict_generator(
        ScoreGen(features, verbose=verbose),
        max_queue_size=8,
        workers=4,
        verbose=0)
    score = score_reshape(score, features)
    return features, score


def make_steps(step, ampl):
    """
    Perform training epochs
    @param step Number of epochs to perform
    @param ampl the K, the randomized component of the score matrix.
    """
    global w2ts, t2i, steps, features, score, histories

    # shuffle the training pictures
    random.shuffle(train)

    # Map whale id to the list of associated training picture hash value
    w2ts = metadata['w2ts']
    # Map training picture hash value to index in 'train' array
    t2i = metadata['t2i']
    # Compute the match score for each picture pair
    features, score = compute_score()

    # Train the model for 'step' epochs
    history = model.fit_generator(
        TrainingData(
            shape,
            score + ampl * np.random.random_sample(size=score.shape),
            steps=step,
            batch_size=32,
            **metadata),
        initial_epoch=steps,
        epochs=steps + step,
        max_queue_size=8,
        workers=4,
        verbose=1).history
    steps += step

    # Collect history data
    history['epochs'] = steps
    history['ms'] = np.mean(score)
    history['lr'] = get_lr(model)
    print(history['epochs'], history['lr'], history['ms'])
    histories.append(history)


histories = []
steps = 0

if isfile('../input/piotte/mpiotte-standard.model'):
    tmp = keras.models.load_model('../input/piotte/mpiotte-standard.model')
    model.set_weights(tmp.get_weights())
else:
    # epoch -> 10
    make_steps(10, 1000)
    ampl = 100.0
    for _ in range(2):
        print('noise ampl.  = ', ampl)
        make_steps(5, ampl)
        ampl = max(1.0, 100**-0.1 * ampl)
    # epoch -> 150
    for _ in range(18):
        make_steps(5, 1.0)
    # epoch -> 200
    set_lr(model, 16e-5)
    for _ in range(10):
        make_steps(5, 0.5)
    # epoch -> 240
    set_lr(model, 4e-5)
    for _ in range(8):
        make_steps(5, 0.25)
    # epoch -> 250
    set_lr(model, 1e-5)
    for _ in range(2):
        make_steps(5, 0.25)
    # epoch -> 300
    weights = model.get_weights()
    model, branch_model, head_model = baseline.build_model(
        shape, 64e-5, 0.0002)
    model.set_weights(weights)
    for _ in range(10):
        make_steps(5, 1.0)
    # epoch -> 350
    set_lr(model, 16e-5)
    for _ in range(10):
        make_steps(5, 0.5)
    # epoch -> 390
    set_lr(model, 4e-5)
    for _ in range(8):
        make_steps(5, 0.25)
    # epoch -> 400
    set_lr(model, 1e-5)
    for _ in range(2):
        make_steps(5, 0.25)
    model.save('../pre_trained_model/standard.model')