
import math
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import multiprocessing as mul
from util import list2libsvm, oversampling
from libsvm.python.plotroc import *
import matplotlib
import sys
from sklearn.metrics import accuracy_score
import numpy as np
matplotlib.use('AGG')
import multiprocessing as mul
from libsvm.python.plotroc import *
import matplotlib
import sys
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
#用于计算测试结果的指标
def performance(label_list, pred_array):

    if len(label_list) != len(pred_array):
        raise ValueError("The number of the original labels must equal to that of the predicted labels.")


    Acc = accuracy_score(np.array(label_list), np.argmax(pred_array, axis=1))
    F = f1_score(np.array(label_list), np.argmax(pred_array, axis=1), average='macro')
    Pre = precision_score(np.array(label_list), np.argmax(pred_array, axis=1), average='macro', zero_division=0)
    Rec = recall_score(np.array(label_list), np.argmax(pred_array, axis=1), average='macro')
    Roc = roc_auc_score(np.array(label_list), pred_array,average='macro',multi_class='ovo')

    return Acc,F,Pre,Rec,Roc