import numpy as np
from scipy.stats import scoreatpercentile
from sklearn.metrics import precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

def get_top_n(roc_list, n, top=True):
    roc_list = np.asarray(roc_list)
    length = roc_list.shape[0]
    roc_sorted = np.partition(roc_list, length - n)
    threshold = roc_sorted[int(length - n)]
    if top:
        return np.where(np.greater_equal(roc_list, threshold))
    else:
        return np.where(np.less(roc_list, threshold))
    
def print_baseline(X_train_new_orig, y, roc_list):
    max_value_idx = roc_list.index(max(roc_list))
    print()
    print('Highest TOS ROC:', roc_list[max_value_idx])
    # normalized score
    X_train_all_norm = StandardScaler().fit_transform(X_train_new_orig)
    X_train_all_norm_mean = np.mean(X_train_all_norm, axis=1)

    roc = np.round(roc_auc_score(y, X_train_all_norm_mean), decimals=4)
    print('Average TOS ROC:', roc)