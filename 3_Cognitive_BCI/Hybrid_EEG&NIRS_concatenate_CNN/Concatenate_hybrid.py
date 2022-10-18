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
from lht_torch import DS_set

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
            torch.nn.Conv2d(40, 40, kernel_size=(1,50), stride=(1,10), dilation=(1, 15))
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
            torch.nn.Conv2d(200, 40, kernel_size=(1,10), stride=(1, 10), dilation=(1, 81), bias=False)
            )
        
        self.layer30 = torch.nn.Sequential(
            torch.nn.Linear(8040, 1024),
            torch.nn.Linear(1024, 256),
            torch.nn.Linear(256, 16),
            torch.nn.Linear(16, 2)
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
        deepfeat = nout
        nout = self.layer26(nout)
        nout = identity(nout)
        nout = self.layer27(nout)
        nout = F.elu(nout, 1)
        nout = self.layer28(nout)
        nout = identity(nout)
        nout = self.layer29(nout)
        nout = np.squeeze(nout)
        
        conout = torch.cat((nout,eout,deepfeat), dim=2)
        conout = conout.view(-1,12080)
        conout = self.layer30(conout)
        
        conout = F.log_softmax(conout, 1)
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
        ratio_tr = np.int8(np.round(np.size(train_index)*0.75))
                
        EEG_X_train, EEG_X_test = EEG_X[train_index], EEG_X[test_index]
        EEG_train = EEG_X_train[0:ratio_tr,:,:,:]
        EEG_val = EEG_X_train[ratio_tr:,:,:,:]
        NIRS_X_train, NIRS_X_test = NIRS_X[train_index], NIRS_X[test_index]
        NIRS_train = NIRS_X_train[0:ratio_tr,:,:,:]
        NIRS_val= NIRS_X_train[ratio_tr:,:,:,:]
        Y_train1, Y_test = Y[train_index], Y[test_index]
        Y_train = Y_train1[0:ratio_tr,:]
        Y_val = Y_train1[ratio_tr:,:]
        
        etrn_loader = DS_set(EEG_train, Y_train)
        ntrn_loader = DS_set(NIRS_train, Y_train)
        eval_loader = DS_set(EEG_val, Y_val)
        nval_loader = DS_set(NIRS_val, Y_val)
        ete_loader = DS_set(EEG_X_test, Y_test)
        nte_loader = DS_set(NIRS_X_test,Y_test)
        
        for i in range(1, training_epochs+1):  
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
                    tr_total_num += label.size(0)
                    
            train_accuracy = correct/tr_total_num*100
            train_Loss.append(train_loss.item())
            train_Acc.append(train_accuracy)
            
            corr = 0
            correct = 0
            val_total_num = 0
            best_misclass = 100
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
                        val_total_num += label1.size(0)
                        
                val_accuracy = correct/val_total_num*100
                val_Loss.append(val_loss.item())
                val_Acc.append(val_accuracy)
                
                if val_Loss[-1] <= best_misclass:
                    best_model = model
                    best_val_loss = val_Loss[-1]
                    best_epoch = training_epochs
                    
                if (i % 20) == 0:
                    print('Epoch: {},  Tr_Acc: {:.5f},  Tr_Loss: {:.5f}'.format(i,train_Acc[-1],train_Loss[-1]))
                    print('Epoch: {},  Val_Acc: {:.5f},  Val_Loss: {:.5f}'.format(i,val_Acc[-1],val_Loss[-1]))
                            
                    
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
            
            print('Epoch: {},  Te_Acc: {:.5f},  Te_Loss: {:.5f}'.format(i,test_Acc[-1],test_Loss[-1]))
                    
        n_sp = n_sp + 1
        del model, train_loss, val_loss, test_loss, train_output, val_output, te_output
                
    Final_Acc[temp_subNum - 1] = (sum(test_Acc) / len(test_Acc))
    del test_Acc, test_Loss, val_Acc, val_Loss, train_Acc, train_Loss, EEG_X, NIRS_X, Y