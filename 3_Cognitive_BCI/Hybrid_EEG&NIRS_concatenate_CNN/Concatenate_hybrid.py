# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 22:59:50 2020

@author: Lee
"""

## import function
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch.nn.functional as F

import os
default_path = 'D:/Project/2020/Deeplearning'

os.chdir(default_path)
del default_path

def identity(x):
    return x

def square(x):
    return x * x

def safe_log(x, eps=1e-6):
    return torch.log(torch.clamp(x, min=eps))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)
    
## Model
class concatenation_CNN(torch.nn.Module):
    def __init__(self):
        super(concatenation_CNN, self).__init__()
        
        ## CNN
        self.layer11 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 40, kernel_size=(1,25), stride=(1,1)),
            torch.nn.Conv2d(40, 40, kernel_size=(10,1), stride=(1,1), padding = 0, bias = False),
            torch.nn.BatchNorm2d(40, eps=1e-05, momentum=0.1, affine = True, track_running_stats=True)
            )
        
        self.layer12 = torch.nn.Sequential(
            torch.nn.AvgPool2d(kernel_size=(1,75), stride=(1, 1), padding=0)
            )
        
        self.layer13 = torch.nn.Sequential(
            torch.nn.Dropout(p=0.5, inplace=False),
            torch.nn.Conv2d(40, 2, kernel_size=(1,50), stride=(1,1), dilation=(1, 15))
            )
        
        ## Deep
        self.layer21 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 25, kernel_size=(1,10), stride=(1,1)),
            torch.nn.Conv2d(25, 25, kernel_size=(18,1), stride=(1,1), bias = False),
            torch.nn.BatchNorm2d(25, eps=1e-05, momentum=0.1, affine = True, track_running_stats=True)
            )
        
        self.layer22 = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=(1,3), stride=(1, 1), padding=0, dilation=(1,1), ceil_mode=False)
            )
        
        self.layer23 = torch.nn.Sequential(
            torch.nn.Dropout(p=0.5, inplace=False),
            torch.nn.Conv2d(25, 50, kernel_size=(1,10), stride=(1,1), dilation=(1, 3), bias=False),
            torch.nn.BatchNorm2d(50, eps=1e-05, momentum=0.1, affine = True, track_running_stats=True)
            )
        
        self.layer24 = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=(1,3), stride=(1, 1), padding=0, dilation=(1,3), ceil_mode=False)
            )
        
        self.layer25 = torch.nn.Sequential(
            torch.nn.Dropout(p=0.5, inplace=False),
            torch.nn.Conv2d(50, 100, kernel_size=(1,10), stride=(1, 1), dilation=(1, 9), bias=False),
            torch.nn.BatchNorm2d(100, eps=1e-05, momentum=0.1, affine = True, track_running_stats=True)
            )
        
        self.layer26 = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=(1,3), stride=(1, 1), padding=0, dilation=(1,9), ceil_mode=False)
            )
        
        self.layer27 = torch.nn.Sequential(
            torch.nn.Dropout(p=0.5, inplace=False),
            torch.nn.Conv2d(100, 200, kernel_size=(1,10), stride=(1, 1), dilation=(1, 27), bias=False),
            torch.nn.BatchNorm2d(200, eps=1e-05, momentum=0.1, affine = True, track_running_stats=True)
            )
        
        self.layer28 = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=(1,3), stride=(1, 1), padding=0, dilation=(1,27), ceil_mode=False)
            )
        
        self.layer29 = torch.nn.Sequential(
            torch.nn.Dropout(p=0.5, inplace=False),
            torch.nn.Conv2d(200, 2, kernel_size=(1,10), stride=(1, 1), dilation=(1, 81), bias=False)
            )
        
        
    def forward(self, x):
        ##CNN
        eeg = x[0]
        nirs = x[1]
        eout = self.layer11(eeg)
        eout = square(eout)
        eout = self.layer12(eout)
        eout = safe_log(eout)
        eout = self.layer13(eout)
        eout = F.log_softmax(eout, 1)
        eout = np.squeeze(eout)
        
        ##DNN
        nout = self.layer21(nirs)
        nout = F.elu(nout, 1)
        nout = self.layer22(nout)
        nout = identity(nout)
        nout = self.layer23(nout)
        nout = F.elu(nout, 1)
        nout = self.layer24(nout)
        nout = identity(nout)
        nout = self.layer25(nout)
        nout = F.elu(nout, 1)
        nout = self.layer26(nout)
        nout = identity(nout)
        nout = self.layer27(nout)
        nout = F.elu(nout, 1)
        nout = self.layer28(nout)
        nout = identity(nout)
        nout = self.layer29(nout)
        nout = F.log_softmax(nout, 1)
        nout = np.squeeze(nout)
        
        #print('EEG feature map size : {}'.format(eout.size()))
        #print('NIRS feature map size : {}'.format(nout.size()))
        conout = torch.cat((nout,eout), dim=2)
        #print('Concatenation feature map size : {}'.format(conout.size()))
        conout = torch.mean(conout, dim=2)
        
        return conout

## classification
Final_Acc = np.zeros(1)
Matrix_Loss = []
Matrix_Acc = []

## define CNN model

training_epochs = 250
batch_size = 64

import warnings
warnings.simplefilter('ignore')

Final_Acc = np.zeros(29)

for temp_subNum in range(1, 30):
    print(temp_subNum)
    
    train_Loss = []
    train_Acc = []
    val_Loss = []
    val_Acc = []
    test_Loss = []
    test_Acc = []
    test_Acc2 = []

    from scipy import io
    raw = io.loadmat('D:/Project/2020/Deeplearning/Hybrid_MA/fv/Shallow/fv_' + str(temp_subNum) + '.mat')
    EEG_X = raw['EEG_X']
    NIRS_X = raw['NIRS_X']
    Y = raw['Y']
    
    ###########################################################################
    ### (3) Preprocessing #####################################################
    ###########################################################################
    from braindecode.datautil.signalproc import exponential_running_standardize
    for ii in range(0, 60):
        # 1. Data reconstruction
        temp_data = EEG_X[ii, :, :]
        temp_data = temp_data.transpose()
        ExpRunStand_data = exponential_running_standardize(temp_data, factor_new=0.001, init_block_size=None, eps=0.0001)
        ExpRunStand_data = ExpRunStand_data.transpose()
        EEG_X[ii, :, :] = ExpRunStand_data
        del temp_data, ExpRunStand_data
        
    for ii in range(0, 60):
        # 1. Data reconstruction
        temp_data = NIRS_X[ii, :, :]
        temp_data = temp_data.transpose()
        ExpRunStand_data = exponential_running_standardize(temp_data, factor_new=0.001, init_block_size=None, eps=0.0001)
        ExpRunStand_data = ExpRunStand_data.transpose()
        NIRS_X[ii, :, :] = ExpRunStand_data
        del temp_data, ExpRunStand_data
        
    from sklearn.model_selection import RepeatedStratifiedKFold
    EEG_X = EEG_X[:,np.newaxis,:,:]
    NIRS_X = NIRS_X[:,np.newaxis,:,:]
    
    rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=1, random_state=36851234)
    n_sp = 1;
    for train_index, test_index in rskf.split(EEG_X, Y):
        # print("TRAIN:", train_index, "TEST:", test_index)
        
        model = concatenation_CNN().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.5*0.001, eps=1e-08, betas=(0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 10)

        EEG_X_train, EEG_X_test = EEG_X[train_index], EEG_X[test_index]
        EEG_train = EEG_X_train[0:43,:,:,:]
        EEG_val = EEG_X_train[43:54,:,:,:]
        NIRS_X_train, NIRS_X_test = NIRS_X[train_index], NIRS_X[test_index]
        NIRS_train = NIRS_X_train[0:43,:,:,:]
        NIRS_val= NIRS_X_train[43:54,:,:,:]
        Y_train1, Y_test = Y[train_index], Y[test_index]
        Y_train = Y_train1[0:43,:]
        Y_val = Y_train1[43:54,:]
        
        EEG_train = torch.tensor(EEG_train, dtype=torch.float32)
        NIRS_train = torch.tensor(NIRS_train, dtype=torch.float32)
        EEG_val = torch.tensor(EEG_val, dtype=torch.float32)
        NIRS_val = torch.tensor(NIRS_val, dtype=torch.float32)
        Y_train = torch.tensor(Y_train, dtype=torch.float32)
        Y_val = torch.tensor(Y_val, dtype=torch.float32)
        
        EEG_train_set = TensorDataset(EEG_train, Y_train)
        EEG_val_set = TensorDataset(EEG_val, Y_val)
        NIRS_train_set = TensorDataset(NIRS_train, Y_train)
        NIRS_val_set = TensorDataset(NIRS_val, Y_val)
        #EEG_train_set, EEG_val_set = torch.utils.data.random_split(edstr, [43, 11])
        #NIRS_train_set, NIRS_val_set = torch.utils.data.random_split(ndstr, [43, 11])
        
        EEG_X_test = torch.tensor(EEG_X_test, dtype=torch.float32)
        NIRS_X_test = torch.tensor(NIRS_X_test, dtype=torch.float32)
        Y_test = torch.tensor(Y_test, dtype=torch.float32)
        edste = TensorDataset(EEG_X_test, Y_test)
        ndste = TensorDataset(NIRS_X_test, Y_test)
        
        etrn_loader = DataLoader(EEG_train_set, batch_size, shuffle=False)
        ntrn_loader = DataLoader(NIRS_train_set, batch_size, shuffle=False)
        eval_loader = DataLoader(EEG_val_set, batch_size, shuffle=False)
        nval_loader = DataLoader(NIRS_val_set, batch_size, shuffle=False)
        ete_loader = DataLoader(edste, batch_size, shuffle=False)
        nte_loader = DataLoader(ndste, batch_size, shuffle=False)
        
        for training_epochs in range(1, training_epochs+1):  
            corr = 0
            correct = 0
            tr_total_num = 0
            for ei,edata in enumerate(etrn_loader):
                for ni,ndata in enumerate(ntrn_loader):
                    ex, label = edata
                    nx, label = ndata
                    ex = ex.to(device)
                    nx = nx.to(device)
                    label = label.squeeze_()
                    label = label.to(device)
                    optimizer.zero_grad()
                    train_output = model([ex,nx])
                    output1 = train_output.argmax(dim=1)
                    train_loss = F.nll_loss(train_output, label.long())
                    corr = label[label == output1.to(device)].size(0)
                    correct += corr
                    train_loss.backward()
                    optimizer.step()
                    scheduler.step()
            train_accuracy = correct/43*100
            train_Loss.append(train_loss.item())
            train_Acc.append(train_accuracy)
            
            corr = 0
            correct = 0
            val_total_num = 0
            best_misclass = 1
            with torch.no_grad():
                for ei,edata1 in enumerate(eval_loader):
                    for ni,ndata1 in enumerate(nval_loader):
                        ex1, label1 = edata1
                        nx1, label1 = ndata1
                        ex1 = ex1.to(device)
                        nx1 = nx1.to(device)
                        label1 = label1.squeeze_()
                        label1 = label1.to(device)
                        val_output = model([ex1, nx1])
                        output2 = val_output.argmax(dim=1)
                        val_loss = F.nll_loss(val_output, label1.long())
                        corr = label1[label1 == output2.to(device)].size(0)
                        correct += corr
                val_accuracy = correct/11*100
                val_Loss.append(val_loss.item())
                val_Acc.append(val_accuracy)
                
                if (1-val_accuracy) <= best_misclass:
                    best_model = model
                    best_val_loss = (1-val_accuracy)
                    
            #if (training_epochs % 250) == 0:
            #    print('[{},{}]Train Loss:{:.4f}, Tr_Accuracy:{:.2f}%'.format(n_sp,training_epochs, train_loss, train_accuracy)) 
            #    print('[{},{}]Validation Loss:{:.4f}, Validation_Accuracy:{:.2f}%'.format(n_sp,training_epochs, val_loss, val_accuracy))
            
        torch.save(best_model.state_dict(), 'D:/Project/2020/Deeplearning/Hybrid_MA/model_save/model_{}_{}.pt'.format(temp_subNum,n_sp))
        
        corr=0
        correct=0
        test_total_num=0
        del model, best_model
        model = concatenation_CNN().to(device)
        model.load_state_dict(torch.load('D:/Project/2020/Deeplearning/Hybrid_MA/model_save/model_{}_{}.pt'.format(temp_subNum,n_sp)))
        with torch.no_grad():
            for ei,edata2 in enumerate(ete_loader):  
                for ni,ndata2 in enumerate(nte_loader):  
                    ex2,label2 = edata2
                    nx2,label2 = ndata2
                    ex2 = ex2.to(device)
                    nx2 = nx2.to(device)
                    label2 = label2.squeeze_()
                    label2 = label2.to(device)
                    te_output = model([ex2, nx2])
                    output3 = te_output.argmax(dim=1)
                    test_loss = F.nll_loss(te_output, label2.long())
                    corr = label2[label2 == output3.to(device)].size(0)
                    correct += corr
                    test_total_num += label2.size(0)
                
            test_accuracy = correct/test_total_num*100
            test_Loss.append(test_loss)
            test_Acc.append(test_accuracy)
            test_Acc2.append(test_accuracy)
            
            #print('[{},{}]Test Loss:{:.4f}, Te_Accuracy:{:.2f}%'.format(n_sp,training_epochs, test_loss, test_accuracy))
        n_sp = n_sp + 1
        del model, train_loss, val_loss, test_loss, train_output, val_output, te_output
                
    Final_Acc[temp_subNum - 1] = (sum(test_Acc) / len(test_Acc))
    del test_Acc, test_Loss, val_Acc, val_Loss, train_Acc, train_Loss, EEG_X, NIRS_X, Y