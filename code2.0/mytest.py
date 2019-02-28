
import sys
import numpy as np
from common import *
from scipy.io import loadmat
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from joint_bayesian import *



def excute_train():

    train_data=
    train_label=
    result_fold= "../result/"

    # pca training.
    pca = PCA(n_components = 2000).fit(train_data)
    data_pca = pca.transform(train_data)
    data_to_pkl(data_pca, result_fold+"pca_train.pkl")
 
    # joint bayes training 
    JointBayesian_Train(data_pca, label, result_fold)


def excute_test():

    test_data =
    test_label = 
    result_fold = "../result/"

    with open(result_fold+"A.pkl", "rb") as f:
        A = pickle.load(f)
    with open(result_fold+"G.pkl", "rb") as f:
        G = pickle.load(f)

    data_pca = pca.transform(test_data)
    data_to_pkl(data_pca, result_fold+"pca_test.pkl")

    # predict using AG
    distance = get_ratios(A, G, test_label, data_pca)
    label = np.repeat(1, len(np.asarray(distance)))

    data_to_pkl({"distance": distance, "label": label}, result_fold+"result.pkl")

if __name__ == "__main__":
    excute_train()
    excute_test()
    excute_performance("../result/result.pkl", -16.9, -16.6, 0.01)
