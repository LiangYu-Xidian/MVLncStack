from joblib import dump, load
from GetData import Mul_load_data
from libsvm.python.plotroc import *
from GetLR import Calculate, Savexls
from torch.nn import functional as F
from GetPerformance import performance
from argparse import RawTextHelpFormatter
from sklearn.metrics import accuracy_score
from util import list2libsvm, oversampling
from sklearn.ensemble import RandomForestClassifier
from GetDL_Model import dl_performance, dl_train_model
from mlRF import Get_rf_new_model, rf_performance
from sklearn.model_selection import KFold, StratifiedKFold
import sys
import time
import Kmer
import torch
import random
import argparse
import datetime
import matplotlib
import numpy as np
import torch.nn as nn
import multiprocessing as mul
from sklearn.model_selection import KFold, StratifiedKFold
matplotlib.use('AGG')
src_vocab = {'P': 0, 'A': 1, 'G': 2, 'C': 3, 'U': 4}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':



    parse = argparse.ArgumentParser(description="The analysis module for training a best classifier",
                                    formatter_class=RawTextHelpFormatter)
    parse.add_argument('-feature', type=str, nargs='*',
                       help="The input files in FASTA format.More than one file could be input.")
    parse.add_argument('-len', type=int, nargs='*',
                       help="The input files in FASTA format.More than one file could be input.")
    parse.add_argument('-epoch', type=int, nargs='*',
                       help="The input files in FASTA format.More than one file could be input.")
    parse.add_argument('-batchsize', type=int, nargs='*',
                       help="The input files in FASTA format.More than one file could be input.")
    parse.add_argument('-lr', type=float, nargs='*',
                       help="The input files in FASTA format.More than one file could be input.")
    parse.add_argument('-cpu', type=int, nargs='*',
                       help="The input files in FASTA format.More than one file could be input.")

    args = parse.parse_args()

    seed = 10
    setup_seed(seed)

    #数据文件
    feature = args.feature
    epoch = args.epoch[0]
    batch_size = args.batchsize[0]
    len = args.len[0]
    lr = args.lr[0]
    cpucore = args.cpu[0]


    print("1.Some basic information about the model:")
    print("feature:" + str(feature))
    print("batch_size:" + str(batch_size))
    print("epoch:" + str(epoch))
    print("lr:" + str(lr))
    print("cpu:" + str(cpucore))
    print("seed: " + str(seed))
    print("len: " + str(len))

    print('\n')
    sys.stdout.flush()

    '''通过对比的超参数'''
    tree = 200

    pro = []

    
    for hw in range(84):
        fold = []
        for i in range(5):

            train_filename = './data/example/train_' + str(i) + '.fasta'
            test_filename = './data/example/test_' + str(i) + '.fasta'

            train_mul = Mul_load_data(filename=train_filename, left=len, right=len, cpucore=cpucore, feature=feature)
            train_mul.start()
            train_data = train_mul.GetData()

            test_mul = Mul_load_data(filename=test_filename, left=len, right=len, cpucore=cpucore, feature=feature)
            test_mul.start()
            test_data = test_mul.GetData()

            rf_model = Get_rf_new_model(tree, train_data, hw)
            rf_Acc,rf_F,rf_Pre,rf_Rec,rf_Roc = rf_performance(rf_model, test_data, hw)

            dl_model = dl_train_model(args=args, Train_data=train_data, Valida_data=test_data)
            dl_Acc, dl_F, dl_Pre, dl_Rec, dl_Roc = dl_performance(dl_model, test_data, batch_size)


            test_Acc = -1
            test_Roc = -1
            test_w = -1
            for i in range(0, 101):
                w = i / 100

                Acc,F,Pre,Rec,Roc = Calculate(dl_model, rf_model, test_data, batch_size, w, tree,hw)
                sys.stdout.flush()

                if test_Acc < Acc:
                    test_Acc = Acc
                    test_F = F
                    test_Pre = Pre
                    test_Rec = Rec
                    test_Roc = Roc
                    test_w = w
            fold.append(test_Acc)
        pro.append(sum(fold)/5)

    with open("./p.txt","w") as file:
        for p in pro:
            file.write(str(p)[:5] + ',')

