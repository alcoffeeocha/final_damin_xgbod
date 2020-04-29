"""
Author: Yue Zhao
Edited by: Gede Agung B., Muh. Alkahfi K. A., dan Muh. Miftah F.
"""

import os
import scipy.io as scio
import numpy as np
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost.sklearn import XGBClassifier
from imblearn.ensemble import BalancedBaggingClassifier
from models.utility import print_baseline
from models.generate_TOS import get_TOS_knn
from models.generate_TOS import get_TOS_loop
from models.generate_TOS import get_TOS_lof
from models.select_TOS import random_select, accurate_select, balance_select

# Memasukan dataset
mat = scio.loadmat(os.path.join('datasets', 'arrhythmia.mat'))
#mat = scio.loadmat(os.path.join('datasets', 'cardio.mat'))
#mat = scio.loadmat(os.path.join('datasets', 'letter.mat'))

X = mat['X']    
y = mat['y']

# Menormalisasikan data X dengan Zscore untuk meningkatkan hasil knn, LoOP, and LOF
scaler = StandardScaler().fit(X)    
X_norm = normalize(X)
F_list = []

# Menjalankan algoritma KNN-base untuk menghasilkan fitur tambahan
# mendefinisikan kembali range k
k_range = [1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50]
# mendefinisikan kembali range k untuk digunakan dengan LoOP disebabkan kompleksitas yang tinggi
k_range_short = [1, 3, 5, 10]

# memvalidasi nilai k
valid = []
for k in k_range:
    if(k < len(X)):
        valid.append(k)
############################################################################

# Menghasilkan TOS dengan algoritma KNN based
F_list, roc_knn, result_knn = get_TOS_knn(X_norm, y, valid, F_list)
# Menghasilkan TOS dengan LoOP
F_list, roc_loop, result_loop = get_TOS_loop(X, y, k_range_short, F_list)
# Menghasilkan TOS dengan LOF
F_list, roc_lof, result_lof = get_TOS_lof(X_norm, y, valid, F_list)
##############################################################################

# Menggabungkan feature space dengan concate beberapa TOS
X_train_new_orig = np.concatenate((result_knn, result_loop, result_lof), axis=1)

X_train_all_orig = np.concatenate((X, X_train_new_orig), axis=1)

# list ROC 
roc_list = roc_knn + roc_loop + roc_lof

# Mengambil hasil baseline
print_baseline(X_train_new_orig, y, roc_list)
##############################################################################
# TOS selection, dengan beberapa metode
jum = 30  # Jumlah TOS yang diselect

# random selection
new_rand_trainX, all_rand_trainX = random_select(X, X_train_new_orig, roc_list, jum)
# accurate selection
new_accu_trainX, all_accu_trainX= accurate_select(X, X_train_new_orig, roc_list, jum)
# balance selection
new_bal_trainX, all_bal_trainX = balance_select(X, X_train_new_orig, roc_list, jum)

###############################################################################
# Membangun beberapa classifier
iterasi = 10 # Jumlah iterasi
ukuran_tes = 0.4  # training = 60%, testing = 40%
hasil = {}

tem_list = [XGBClassifier(), LogisticRegression(penalty="l1",solver='liblinear'),LogisticRegression(penalty="l2",solver='liblinear')]

name_list = ['xgb', 'lr1', 'lr2']
# Menginisiasikan hasil dalam bentuk dict
for name in name_list:
    hasil[name + 'ROC' + 'o'] = []
    hasil[name + 'ROC' + 's'] = []
    hasil[name + 'ROC' + 'n'] = []

for i in range(iterasi):
    ori_len = X.shape[1]
    
    #menggunakan semua TOS
    #trainX, testX, train_y, test_y = train_test_split(X_train_all_orig, y, test_size=ukuran_tes)
    # Menggunakan random selection
    #trainX, testX, train_y, test_y = train_test_split(all_rand_trainX, y, test_size=ukuran_tes)
    # Menggunakan accurate selection
    #trainX, testX, train_y, test_y = train_test_split(all_accu_trainX, y, test_size=ukuran_tes)
    # Menggunakan balance selection
    trainX, testX, train_y, test_y = train_test_split(all_bal_trainX, y, test_size=ukuran_tes)
    # Memakai original feature
    trainX_o = trainX[:, 0:ori_len]
    testX_o = testX[:, 0:ori_len]

    trainX_n = trainX[:, ori_len:]
    testX_n = testX[:, ori_len:]

    for tamp, name in zip(tem_list, name_list):
        print('Processing', name, 'round', i + 1)
        if name != 'xgb':
            tamp = BalancedBaggingClassifier(base_estimator=tamp, replacement=False)
        
        # fully supervised
        tamp.fit(trainX_o, train_y.ravel())
        y_pred = tamp.predict_proba(testX_o)

        scoreROC = roc_auc_score(test_y, y_pred[:, 1])
        hasil[name + 'ROC' + 'o'].append(scoreROC)
        
        # unsupervised
        tamp.fit(trainX_n, train_y.ravel())
        y_pred = tamp.predict_proba(testX_n)

        scoreROC = roc_auc_score(test_y, y_pred[:, 1])
        hasil[name + 'ROC' + 'n'].append(scoreROC)
        
        # semi-supervised
        tamp.fit(trainX, train_y.ravel())
        y_pred = tamp.predict_proba(testX)

        scoreROC = roc_auc_score(test_y, y_pred[:, 1])
        hasil[name + 'ROC' + 's'].append(scoreROC)

for val in ['ROC']:
    print("="*50)
    for name in name_list:
        print(np.round(np.mean(hasil[name+val+'o']),decimals=5),val,name,'Original Features')
        print(np.round(np.mean(hasil[name+val+'n']),decimals=5),val,name,'TOS Only')
        print(np.round(np.mean(hasil[name+val+'s']),decimals=5),val,name,'Original Feature + TOS')