
import sys
import numpy as np
from common import *
from scipy.io import loadmat
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from joint_bayesian import *

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

def excute_test(test_data, test_label):

    global pca

    result_fold = "../result/"
    
    with open(result_fold+"A.pkl", "rb") as f:
        A = pickle.load(f)
    with open(result_fold+"G.pkl", "rb") as f:
        G = pickle.load(f)

    data_pca = pca.transform(test_data)
    data_to_pkl(data_pca, result_fold+"pca_test.pkl")

    # FIXME !
    scorelist = []
    numlist = len(test_label)
    sacrate = 0
    for teli in range(1: numlist+1):
        acnum = 0
        for teli2 in range(1: numlist+1):
            score = Verify(A, G, data_pca[teli], data_pca[teli2])
            sl = [score, test_label[teli2]]
            scorelist.append(sl)
            if test_label[teli] = test_label[teli2]:
                acnum = acnum + 1
        sorted(scorelist, key=(lambda x:x[0]), reverse=True)
        label_selct = scorelist[i[1] for i in range(0: 5)]
        print (teli)
        print (label_selct)
        acrate = float(acnum) / numlist
        sacrate = sacrate + acrate
    facrate = sacrate/numlist
    print(facrate)

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
