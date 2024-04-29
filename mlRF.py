import sys
import numpy as np
import multiprocessing as mul
import matplotlib
from libsvm.python.plotroc import *
from GetPerformance import performance
from sklearn.metrics import accuracy_score
from util import list2libsvm, oversampling
from sklearn.ensemble import RandomForestClassifier

matplotlib.use('AGG')

def rf_performance(rf_model,Test_data,i):

    test_vector_list = []
    test_label_list = []

    for data in Test_data:
        test_vector_list.append(data.mlfeature[:144+i*16])
        test_label_list.append(int(data.label))

    p_val = rf_model.predict_proba(test_vector_list).tolist()

    #3.评价
    Acc,F,Pre,Rec,Roc = performance(test_label_list, p_val)

    return Acc,F,Pre,Rec,Roc


def Get_rf_new_model(tree,Train_Data_All,i):
    train_vector_list = []
    train_label_list = []
    for data in Train_Data_All:
        train_vector_list.append(data.mlfeature[:144+i*16])
        train_label_list.append(int(data.label))


    train_vector_list, train_label_list = oversampling(train_vector_list, train_label_list)
    rf_model = RandomForestClassifier(n_estimators=tree, random_state=10)
    rf_model.fit(train_vector_list, train_label_list)
    return rf_model