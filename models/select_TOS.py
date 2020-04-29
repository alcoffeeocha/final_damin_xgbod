import random
import numpy as np
from scipy.stats import pearsonr
from models.utility import get_top_n

def random_select(X, trainX_new_ori, roc_list, p):
    F_rand_s = random.sample(range(0, len(roc_list)), p)
    trainX_new_rand = trainX_new_ori[:, F_rand_s]
    trainX_all_rand = np.concatenate((X, trainX_new_rand), axis=1)
    return trainX_new_rand, trainX_all_rand

def accurate_select(X, trainX_new_ori, roc_list, p):
    F_accu_s = get_top_n(roc_list=roc_list, n=p, top=True)
    trainX_new_accu = trainX_new_ori[:, F_accu_s[0][0:p]]
    trainX_all_accu = np.concatenate((X, trainX_new_accu), axis=1)
    return trainX_new_accu, trainX_all_accu
    
def balance_select(X, trainX_new_ori, roc_list, p):
    F_balance_s = []
    pearson_list = np.zeros([len(roc_list), 1])
    # handle the first value
    max_value_idx = roc_list.index(max(roc_list))
    F_balance_s.append(max_value_idx)
    roc_list[max_value_idx] = -1
    for i in range(p - 1):
        for j in range(len(roc_list)):
            pear = pearsonr(trainX_new_ori[:, max_value_idx],trainX_new_ori[:, j])
            # update the pearson
            pearson_list[j] = np.abs(pearson_list[j]) + np.abs(pear[0])
        discounted_roc = np.true_divide(roc_list, pearson_list.transpose())
        max_value_idx = np.argmax(discounted_roc)
        F_balance_s.append(max_value_idx)
        roc_list[max_value_idx] = -1
    X_train_new_balance = trainX_new_ori[:, F_balance_s]
    X_train_all_balance = np.concatenate((X, X_train_new_balance), axis=1)
    return X_train_new_balance, X_train_all_balance
