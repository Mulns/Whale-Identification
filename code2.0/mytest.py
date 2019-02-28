
import sys
import numpy as np
from common import *
from scipy.io import loadmat
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from joint_bayesian import *
import os
pca = PCA(n_components = 180)

def excute_train(train_data, label):

    global pca

    result_fold= "../result/"

    # pca training.
    pca = pca.fit(train_data)
    a = 2
    data_pca = pca.transform(train_data)
    data_to_pkl(data_pca, result_fold+"pca_train.pkl")
 
    # joint bayes training 
    JointBayesian_Train(data_pca, label, result_fold)

def partition(lst, partition_size):
    if partition_size < 1:
        partition_size = 1
    return [
        lst[i:i + partition_size]
        for i in range(0, len(lst), partition_size)
    ]

def excute_test(test_data, test_label):

    global pca

    result_fold = "../result/"
    test_data_folder = ""
    
    with open(result_fold+"A.pkl", "rb") as f:
        A = pickle.load(f)
    with open(result_fold+"G.pkl", "rb") as f:
        G = pickle.load(f)

    data_pca = pca.transform(test_data)
    data_to_pkl(data_pca, result_fold+"pca_test.pkl")

    # FIXME !
    test_list = os.listdir(test_data_folder)
    pair_list = partition(test_list, 2) 

    # predict using AG
    distance = get_ratios(A, G, pair_list, data_pca)
    label = np.repeat(1, len(np.asarray(distance)))

    data_to_pkl({"distance": distance, "label": label}, result_fold+"result.pkl")

if __name__ == "__main__":
    import pickle
    meta_dir = "../Dataset/metadata/"
    with open(meta_dir+"train_embed.pickle", 'rb') as f:
        train_embed = pickle.load(f)
    with open(meta_dir+"train_label.pickle", 'rb') as f:
        train_label = pickle.load(f)
    with open(meta_dir+"test_embed.pickle", 'rb') as f:
        test_embed = pickle.load(f)
    with open(meta_dir+"test_label.pickle", 'rb') as f:
        test_label = pickle.load(f)

    excute_train(train_embed, train_label)
    excute_test(test_embed, test_label)
    excute_performance("../result/result.pkl", -16.9, -16.6, 0.01)
