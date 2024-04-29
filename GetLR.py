import os, random, math, time, sys
import multiprocessing as mul
from itertools import combinations
import pickle
from GetPerformance import performance
from sklearn.linear_model import LogisticRegression

import xlwt
from util import libsvm2list, list2libsvm, oversampling
from GetPerformance import performance
from util import check_contain_chinese, del_file, copy_file, cda, OET_KNN, list2libsvm, libsvm2list, oversampling

from libsvm.checkdata import check_data
from libsvm.python.svmutil import *
from libsvm.python.plotroc import *
from sklearn.metrics import roc_auc_score, roc_curve

import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torch.utils.data as Data
import torch
import numpy
from libsvm.python.plotroc import *
from util import list2libsvm
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab = {0:'P', 1:'A', 2:'G', 3:'C', 4:'U'}

def Calculate(dl_model,rf_model,Test_data,batch_size,w , tree ,i):

    Test_ml = []
    Test_dl = []
    Test_label = []
    for data in Test_data:
        Test_ml.append(data.mlfeature[:144+i*16])
        Test_dl.append(data.dlfeature)
        Test_label.append(int(data.label))

    Test_ml, Test_dl, Test_label = torch.tensor(Test_ml, dtype=torch.float32), torch.tensor(Test_dl, dtype=torch.float32).long(),torch.LongTensor(Test_label)

    Test_dataset = Data.TensorDataset(Test_ml, Test_dl,  Test_label)
    Test_loader = torch.utils.data.DataLoader(
        Test_dataset,
        batch_size=batch_size,
    )

    model = dl_model.eval()
    label_list = []
    pred_list = []

    with torch.no_grad():
        for test_ml, test_dl, test_y in Test_loader:

            test_dl, test_y = test_dl.to(device), test_y.to(device)
            result = model(test_dl)

            dl_pro_val = []
            for data in result.data:
                probability = torch.softmax(data, dim=0)
                probability = probability.cpu().numpy().tolist()
                dl_val = [i*(1-w) for i in probability ]
                dl_pro_val.append(dl_val)


            rf_p_val = rf_model.predict_proba(test_ml)
            rf_pro_val = []
            for val in rf_p_val:
                rf_val = [i*(w) for i in val]
                rf_pro_val.append(rf_val)

            dl_val = numpy.array(dl_pro_val)
            ml_val = numpy.array(rf_pro_val)
            pro_result = dl_val + ml_val

            for label in test_y.tolist(): label_list.append(label)
            for pred in pro_result: pred_list.append(pred.tolist())
        Acc, F, Pre, Rec, Roc = performance(label_list, pred_list)


    return Acc,F,Pre,Rec,Roc


def Savexls(dl_model, rf_model, Test_data, batch_size, w , tree, ran, t):

    Test_name = []
    Test_dl = []
    Test_ml = []
    Test_label = []
    for data in Test_data:
        Test_name.append(data.name)
        Test_dl.append(data.dlfeature)
        Test_ml.append(data.mlfeature)
        Test_label.append(int(data.label))

    model = dl_model.eval()
    Test_ml, Test_dl,Test_label = torch.tensor(Test_ml, dtype=torch.float32), torch.tensor(Test_dl, dtype=torch.float32).long(),torch.LongTensor(Test_label)

    name_list = []
    dl_list = []
    ml_list = []
    label_list = []
    pred_list = []
    true_list = []
    for i in range(len(Test_ml)):

        test_dl = Test_dl[i]


        test_dl = torch.unsqueeze(test_dl,0).to(device)

        result = model(test_dl)

        dl_pre_val = []
        dl_orign_list = []
        for data in result.data:
            probability = torch.softmax(data, dim=0)
            probability = probability.cpu().numpy().tolist()
            dl_orign = [i for i in probability]
            dl_orign_list.append(dl_orign)
            dl_val = [i * (1 - w) for i in probability]
            dl_pre_val.append(dl_val)


        test_ml = np.array(test_ml)[np.newaxis,:].tolist()

        rf_p_val = rf_model.predict_proba(test_ml)
        rf_pre_val = []
        ml_orign_list = []
        for val in rf_p_val:
            ml_orign = [i for i in val]
            ml_orign_list.append(ml_orign)
            rf_val = [i * (w) for i in val]
            rf_pre_val.append(rf_val)

        dl_val = numpy.array(dl_pre_val)
        ml_val = numpy.array(rf_pre_val)
        pred_result = dl_val + ml_val


        name_list.append(Test_name[i])
        dl_list.append(dl_val[0])
        ml_list.append(ml_val[0])
        pred_list.append(pred_result[0])
        label_list.append(Test_label[i])


    #     Acc, F, Pre, Rec, Roc = performance(label_list, pred_list)
    #
    # print('Acc: ' + str(Acc) + ' Roc: ' + str(Roc) + ' F: ' + str(F) + ' Pre: ' + str(Pre) + ' Rec: ' + str(Rec) + ' w: ' + str(w))

    length = len(dl_list)
    book = xlwt.Workbook(encoding='utf-8',style_compression=0)
    sheet = book.add_sheet('OUTPUT',cell_overwrite_ok=True)
    col = ('Name','DL_Pro','DL_Label','RF_Pro','RF_Label','STACK_Pro','Predict','Label','w')
    for i in range(len(col)):
        sheet.write(0,i,col[i])

    predict_list = np.argmax(pred_list, axis=1)
    dl_label_list = np.argmax(dl_list, axis=1)
    rf_label_list = np.argmax(ml_list, axis=1)

    subcellular_list = ['Nucleus','Cytoplasm','Ribosome','Exosome']

    count = 0
    for i in range(length):
        data_list = []
        data_list.append(name_list[i])
        data_list.append(str(round(dl_list[i][label_list[i]],3)))
        data_list.append(str(subcellular_list[dl_label_list.tolist()[i]]))
        data_list.append(str(round(ml_list[i][label_list[i]],3)))
        data_list.append(str(subcellular_list[rf_label_list.tolist()[i]]))
        data_list.append(str(round(pred_list[i][label_list[i]],3)))
        data_list.append(str(subcellular_list[predict_list.tolist()[i]]))
        data_list.append(str(subcellular_list[label_list[i]]))
        data_list.append(str(w))


        count += 1
        for j in range(len(data_list)):
            sheet.write(count,j,data_list[j])
    book.save('./Predict/All_' + str(tree) + '_' + str(ran) + '_' + str(t) + '.xls')

    file = open('./Predict/predict_' + str(tree) + '_' + str(ran) + '_' + str(t) + '.txt', 'w')
    file.write('Deep Learning:\n')
    for dl in dl_list:
        file.write(str(dl)+'\n')
    file.write('\nRandom Forest:\n')
    for ml in ml_list:
        file.write(str(ml)+'\n')
    file.write('\nStacking:\n')
    for p in pred_list:
        file.write(str(p)+'\n')
    file.write('\nLabel:\n')
    for label in label_list:
        file.write(str(label)+'\n')
