import os
import numpy as np
from tensorflow import keras
from skimage.filters import gaussian
from scipy.misc import imresize, imsave
import image_utils as iu
from PIL import Image
import random
from tqdm import tqdm
import time
import image_utils as iu
# from keras.utils.data_utils import Sequence
from lap import lapjv

TRAIN_DF = '../Dataset/train.csv'
SUB_Df = '../Dataset/sample_submission.csv'
TRAIN = '../Dataset/train/'
TEST = '../Dataset/test/'
P2H = '../Dataset/metadata/p2h.pickle'
P2SIZE = '../Dataset/metadata/p2size.pickle'
H2P = "../Dataset/metadata/h2p.pickle"
W2HS = "../Dataset/metadata/w2hs.pickle"
H2WS = "../Dataset/metadata/w2hs.pickle"
W2TS = "../Dataset/metadata/w2ts.pickle"
T2I = "../Dataset/metadata/t2i.pickle"
TRAIN_ID = "../Dataset/metadata/train_id.pickle"
BB_DF = "../Dataset/metadata/bounding_boxes.csv"

# META_LIST = [P2H, P2SIZE, H2P, W2HS, W2TS, H2WS, T2I, TRAIN_ID]
# p2h, p2size, h2p, w2hs, w2ts, h2ws, t2i, train = iu.load_meta()


class TrainingData(keras.utils.Sequence):
    def __init__(self, shape, score, steps=1000, batch_size=32, **metadata):
        """
        @param score the cost matrix for the picture matching
        @param steps the number of epoch we are planning with this score matrix
        """
        super(TrainingData, self).__init__()
        self.img_shape = shape
        self.score = -score  # Maximizing the score is the same as minimuzing -score.
        self.steps = steps
        self.batch_size = batch_size
        self.w2ts = metadata['w2ts']
        self.t2i = metadata['t2i']
        self.train = metadata['train']
        self.metadata = metadata
        for ts in self.w2ts.values():
            idxs = [self.t2i[t] for t in ts]
            for i in idxs:
                for j in idxs:
                    # Set a large value for matching whales -- eliminates this potential pairing
                    self.score[i, j] = 10000.0
        self.on_epoch_end()

    def __getitem__(self, index):
        start = self.batch_size * index
        end = min(start + self.batch_size, len(self.match) + len(self.unmatch))
        size = end - start
        assert size > 0
        a = np.zeros((size, ) + self.img_shape, dtype=keras.backend.floatx())
        b = np.zeros((size, ) + self.img_shape, dtype=keras.backend.floatx())
        c = np.zeros((size, 1), dtype=keras.backend.floatx())
        j = start // 2
        for i in range(0, size, 2):
            a[i, :, :, :] = iu.read_cropped_image(
                self.match[j][0],
                self.img_shape,
                augment=True,
                **self.metadata)
            b[i, :, :, :] = iu.read_cropped_image(
                self.match[j][1],
                self.img_shape,
                augment=True,
                **self.metadata)
            c[i, 0] = 1  # This is a match
            a[i + 1, :, :, :] = iu.read_cropped_image(
                self.unmatch[j][0],
                self.img_shape,
                augment=True,
                **self.metadata)
            b[i + 1, :, :, :] = iu.read_cropped_image(
                self.unmatch[j][1],
                self.img_shape,
                augment=True,
                **self.metadata)
            c[i + 1, 0] = 0  # Different whales
            j += 1
        return [a, b], c

    def on_epoch_end(self):
        if self.steps <= 0: return  # Skip this on the last epoch.
        self.steps -= 1
        self.match = []
        self.unmatch = []
        t1 = time.time()
        _, _, x = lapjv(self.score)  # Solve the linear assignment problem
        print("lapjv takes %.3f" % (time.time() - t1))
        y = np.arange(len(x), dtype=np.int32)

        # Compute a derangement for matching whales
        for ts in self.w2ts.values():
            d = ts.copy()
            while True:
                random.shuffle(d)
                if not np.any(ts == d): break
            for ab in zip(ts, d):
                self.match.append(ab)

        # Construct unmatched whale pairs from the LAP solution.
        for i, j in zip(x, y):
            if i == j:
                print(self.score)
                print(x)
                print(y)
                print(i, j)
            assert i != j
            self.unmatch.append((self.train[i], self.train[j]))

        # Force a different choice for an eventual next epoch.
        self.score[x, y] = 10000.0
        self.score[y, x] = 10000.0
        random.shuffle(self.match)
        random.shuffle(self.unmatch)
        # print(len(self.match), len(train), len(self.unmatch), len(train))
        assert len(self.match) == len(self.train) and len(self.unmatch) == len(
            self.train)

    def __len__(self):
        return (len(self.match) + len(self.unmatch) + self.batch_size -
                1) // self.batch_size


# A Keras generator to evaluate only the BRANCH MODEL
class FeatureGen(keras.utils.Sequence):
    def __init__(self, data, shape, batch_size=64, verbose=1, **metadata):
        super(FeatureGen, self).__init__()
        self.data = data
        self.img_shape = shape
        self.batch_size = batch_size
        self.verbose = verbose
        if self.verbose > 0:
            self.progress = tqdm(total=len(self), desc='Features')
        self.metadata = metadata

    def __getitem__(self, index):
        start = self.batch_size * index
        size = min(len(self.data) - start, self.batch_size)
        a = np.zeros((size, ) + self.img_shape, dtype=keras.backend.floatx())
        for i in range(size):
            a[i, :, :, :] = iu.read_cropped_image(
                self.data[start + i],
                self.img_shape,
                augment=False,
                **self.metadata)
        if self.verbose > 0:
            self.progress.update()
            if self.progress.n >= len(self): self.progress.close()
        return a

    def __len__(self):
        return (len(self.data) + self.batch_size - 1) // self.batch_size


class ScoreGen(keras.utils.Sequence):
    def __init__(self, x, y=None, batch_size=2048, verbose=1):
        super(ScoreGen, self).__init__()
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.verbose = verbose
        if y is None:
            self.y = self.x
            self.ix, self.iy = np.triu_indices(x.shape[0], 1)
        else:
            self.iy, self.ix = np.indices((y.shape[0], x.shape[0]))
            self.ix = self.ix.reshape((self.ix.size, ))
            self.iy = self.iy.reshape((self.iy.size, ))
        self.subbatch = (len(self.x) + self.batch_size - 1) // self.batch_size
        if self.verbose > 0:
            self.progress = tqdm(total=len(self), desc='Scores')

    def __getitem__(self, index):
        start = index * self.batch_size
        end = min(start + self.batch_size, len(self.ix))
        a = self.y[self.iy[start:end], :]
        b = self.x[self.ix[start:end], :]
        if self.verbose > 0:
            self.progress.update()
            if self.progress.n >= len(self): self.progress.close()
        return [a, b]

    def __len__(self):
        return (len(self.ix) + self.batch_size - 1) // self.batch_size


if __name__ == "__main__":

    # Test on a batch of 32 with random costs.
    meta_dic = iu.load_meta()
    train = meta_dic['train']
    score = np.random.random_sample(size=(len(train), len(train)))
    data = TrainingData((384, 384, 1), score, **meta_dic)
    # (a, b), c = data[0]

    for d in data:
        (a, b), c = d
        print(a.shape)
        print(b.shape)
        print(c)
        time.sleep(5)
