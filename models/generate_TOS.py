import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import LocalOutlierFactor
from PyNomaly import loop

def knn(X, n_neighbors):
    neigh = NearestNeighbors()
    neigh.fit(X)

    res = neigh.kneighbors(n_neighbors=n_neighbors, return_distance=True)
    # k-average, k-median, knn
    return np.mean(res[0], axis=1), np.median(res[0], axis=1), res[0][:, -1]

def get_TOS_knn(X, y, k_list, feature_list):
    knn_list = ["Knn_mean", "Knn_median", "Knn_kth"]

    hasil_knn = np.zeros([X.shape[0], len(k_list) * len(knn_list)])
    roc_knn = []
    prec_knn = []

    for i in range(len(k_list)):
        k = k_list[i]
        k_mean, k_median, k_k = knn(X, n_neighbors=k)
        knn_result = [k_mean, k_median, k_k]

        for j in range(len(knn_result)):
            score_pred = knn_result[j]
            clf = knn_list[j]

            roc = np.round(roc_auc_score(y, score_pred), decimals=4)
            print('{clf} #{k} - ROC: {roc} '.format(clf=clf, k=k, roc=roc))
            feature_list.append(clf + str(k))
            roc_knn.append(roc)
            hasil_knn[:, i * len(knn_result) + j] = score_pred

    print()
    return feature_list, roc_knn, hasil_knn

def get_TOS_loop(X, y, k_list, feature_list):
    # only compatible with pandas
    df_X = pd.DataFrame(X)
    result_loop = np.zeros([X.shape[0], len(k_list)])
    roc_loop = []
   
    for i in range(len(k_list)):
        k = k_list[i]
        clf = loop.LocalOutlierProbability(df_X, n_neighbors=k).fit()
        score_pred = clf.local_outlier_probabilities.astype(float)
        roc = np.round(roc_auc_score(y, score_pred), decimals=4)
        print('Iterasi ke-{k} - ROC: {roc}'.format(k=k,roc=roc))

        feature_list.append('loop_' + str(k))
        roc_loop.append(roc)
        result_loop[:, i] = score_pred
    print()
    return feature_list, roc_loop, result_loop

def get_TOS_lof(X, y, k_list, feature_list):
    result_lof = np.zeros([X.shape[0], len(k_list)])
    roc_lof = []
    prec_lof = []

    for i in range(len(k_list)):
        k = k_list[i]
        clf = LocalOutlierFactor(n_neighbors=k)
        y_pred = clf.fit_predict(X)
        score_pred = clf.negative_outlier_factor_

        roc = np.round(roc_auc_score(y, score_pred * -1), decimals=4)
        print('LOF #{k} - ROC: {roc} '.format(k=k,roc=roc))

        feature_list.append('lof_' + str(k))
        roc_lof.append(roc)
       
        result_lof[:, i] = score_pred * -1
    print()
    return feature_list, roc_lof,  result_lof