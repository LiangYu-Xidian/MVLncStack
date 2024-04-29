from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset
# from  torchsampler import ImbalancedDatasetSampler
import torch.optim as optim
import torch.utils.data as Data
import torch.nn as nn
import torch
from GetModel import TextCNN
from GetPerformance import performance
import sys
import numpy as np
from collections import Counter
from torch.nn import functional as F

dtype = torch.FloatTensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vocab = {0:'P', 1:'A', 2:'G', 3:'C', 4:'U'}

#------------------------------------------------------------------------------------------------
#训练深度学习的模型
def dl_train_model(args,Train_data,Valida_data):

    length = args.len[0]
    epoch = args.epoch[0]
    batch_size = args.batchsize[0]

    '''
    1.这里用于处理数据集不平衡的问题
    '''
    Train_dl = []
    Train_label = []
    for data in Train_data:
        Train_dl.append(data.dlfeature)

        if data.label == 0 : Train_label.append([1, 0, 0, 0])
        elif data.label == 1 : Train_label.append([0, 1, 0, 0])
        elif data.label == 2: Train_label.append([0, 0, 1, 0])
        elif data.label == 3: Train_label.append([0, 0, 0, 1])


    input_dl_train, input_label_train = torch.tensor(Train_dl, dtype=torch.float32).long(), torch.tensor(Train_label, dtype=torch.float32)

    Train_dataset = Data.TensorDataset(input_dl_train, input_label_train)
    train_loader = torch.utils.data.DataLoader(
        Train_dataset,
        #sampler=ImbalancedDatasetSampler(Train_dataset),
        batch_size=batch_size,
    )


    Valida_dl = []
    Valida_label = []

    for data in Valida_data:

        Valida_dl.append(data.dlfeature)
        if data.label == 0 : Valida_label.append([1, 0, 0, 0])
        elif data.label == 1 : Valida_label.append([0, 1, 0, 0])
        elif data.label == 2: Valida_label.append([0, 0, 1, 0])
        elif data.label == 3: Valida_label.append([0, 0, 0, 1])

    valida_dl, valida_label = torch.tensor(Valida_dl, dtype=torch.float32).long(), torch.tensor(Valida_label, dtype=torch.float32)
    Valida_dataset = Data.TensorDataset(valida_dl, valida_label)
    valida_loader = torch.utils.data.DataLoader(
        Valida_dataset,
        batch_size=batch_size,
    )


    '''
    2.这里用于通过训练数据集来进行模型的训练
    model = svm_train(train_label_list, train_vector_list, svm_params)
    '''
    model = TextCNN(args).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, 'min')

    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name)
    # print('\n')
    # sys.stdout.flush()
    # exit()

    for epoch_num in range(epoch):
        model = model.train()

        times_of_epoch = 0
        loss_sum = 0

        for batch_dl, batch_y in train_loader:

            times_of_epoch = times_of_epoch + 1
            batch_dl, batch_y = batch_dl.to(device),batch_y.to(device)
            batch_graph = []

            predict = model(batch_dl)
            loss = criterion(predict, batch_y)

            loss_sum = loss_sum + loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_average = float(loss_sum/times_of_epoch)

        '''
        3.这里进行对测试集进行验证
        '''
        model = model.eval()
        times_of_valida_epoch = 0
        valida_loss_sum = 0

        with torch.no_grad():
            for valida_dl, valida_y in valida_loader:
                times_of_valida_epoch = times_of_valida_epoch+1
                valida_dl,valida_y = valida_dl.to(device),valida_y.to(device)
                valida_graph = []

                valida_pred = model(valida_dl)
                valida_loss = criterion(valida_pred,valida_y)
                valida_loss_sum = valida_loss_sum + valida_loss.item()

        valida_average_loss = float(valida_loss_sum/times_of_valida_epoch)
        scheduler.step(valida_average_loss)
        #print('Epoch:', '%04d' % (epoch_num+1), 'Train loss =', '{:.4f}'.format(loss_average),' Valida loss =', '{:.4f}'.format(valida_average_loss))
        sys.stdout.flush()

    return model


def dl_performance(dl_model,Test_data, batch_size):

    Test_dl = []
    Test_label = []
    for data in Test_data:
        Test_dl.append(data.dlfeature)
        Test_label.append(int(data.label))


    Test_dl, Test_label = torch.tensor(Test_dl, dtype=torch.float32).long(),  torch.LongTensor(Test_label)

    Test_dataset = Data.TensorDataset(Test_dl, Test_label)
    Test_loader = torch.utils.data.DataLoader(
        Test_dataset,
        batch_size=batch_size,
    )

    model = dl_model.eval()
    test_val = []
    test_predict_label = []
    test_origin_label = []

    with torch.no_grad():
        for test_dl , test_y in Test_loader:

            test_dl, test_y = test_dl.to(device), test_y.to(device),

            test_pred = model(test_dl)

            test_val_batch = []
            test_label_batch = []

            for data in test_pred.data:
                probability = torch.softmax(data, dim=0)
                probability = probability.cpu().numpy().tolist()
                test_val_batch.append(probability)

            test_val = test_val + test_val_batch
            test_origin_label = test_origin_label + test_y.tolist()

    Acc,F,Pre,Rec,Roc = performance(test_origin_label, test_val)

    return Acc,F,Pre,Rec,Roc
