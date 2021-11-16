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
default_path = 'D:/Project/2021/DL'

os.chdir(default_path)
del default_path

def square(x):
    return x * x

def safe_log(x, eps=1e-6):
    return torch.log(torch.clamp(x, min=eps))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)
    
## Model
class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 40, kernel_size=(1,25), stride=(1,1))
            )
        
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(40, 40, kernel_size=(10,1), stride=(1,1), padding = 0, bias = False)
            )
        
        self.layer3 = torch.nn.Sequential(
            torch.nn.BatchNorm2d(40, eps=1e-05, momentum=0.1, affine = True, track_running_stats=True)
            )
        
        self.layer4 = torch.nn.Sequential(
            torch.nn.AvgPool2d(kernel_size=(1,75), stride=(1, 1), padding=0)
            )
        
        self.layer5 = torch.nn.Sequential(
            torch.nn.Dropout(p=0.5, inplace=False),
            torch.nn.Conv2d(40, 2, kernel_size=(1,50), stride=(1,1), dilation=(1, 15))
            )
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = square(out)
        out = self.layer4(out)
        out = safe_log(out)
        out = self.layer5(out)
        out = F.log_softmax(out, 1)
        out = np.squeeze(out)
        out2 = out
        out = torch.mean(out, dim=2)
        return out, out2

## classification
Final_Acc = np.zeros(1)
Matrix_Loss = []
Matrix_Acc = []

## define CNN model

training_epochs = 250
batch_size = 64

import warnings
warnings.simplefilter('ignore')

###############################################################################
### (2) Load data #############################################################
###############################################################################
# set up data 
import os
default_path = 'D:/Project/2020/Deeplearning'

os.chdir(default_path)
del default_path

nFold = 5;
nShift = 1;

Final_Acc = np.zeros(29)
Fold_Acc = np.zeros(nFold)
Shift_Acc = np.zeros(nShift)
K = np.array(range(1,30))
K = np.delete(K,[15,20])

n = 0
for temp_subNum in range(1, 30):
#for temp_subNum in K:
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
    X = raw['EEG_X']
    Y = raw['Y']
            
    X = X[:,np.newaxis,:,:]
    
    ind = io.loadmat('D:/Project/2021/DL/Brain_Switch/Data/indice2_5fold.mat')
    indice = ind['indice']
    
    for shift in range(0,nShift):
        for fold in range(1,nFold+1):
            test_index = (indice[:,shift+1] == fold)
            train_index = ~test_index
            
            model = CNN().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.5*0.001, eps=1e-08, betas=(0.9, 0.999))
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 10)
    
            X_train, X_test = X[train_index], X[test_index]
            Y_train, Y_test = Y[train_index], Y[test_index]
            
            X_train = torch.tensor(X_train, dtype=torch.float32)
            Y_train = torch.tensor(Y_train, dtype=torch.float32)
            dstr = TensorDataset(X_train, Y_train)
            
            X_test = torch.tensor(X_test, dtype=torch.float32)
            Y_test = torch.tensor(Y_test, dtype=torch.float32)
            dste = TensorDataset(X_test, Y_test)
            
            trn_loader = DataLoader(dstr, batch_size, shuffle=False)
            te_loader = DataLoader(dste, batch_size, shuffle=False)
            
            num_epoch = 0
            for training_epochs in range(1, training_epochs+1):
                num_epoch += 1 
                corr = 0
                correct = 0
                tr_total_num = 0
                for i,data in enumerate(trn_loader):
                    x, label = data
                    x = x.to(device)
                    label = label.squeeze_()
                    label = label.to(device)
                    optimizer.zero_grad()
                    train_output, a = model(x)
                    output1 = train_output.argmax(dim=1)
                    train_loss = F.nll_loss(train_output, label.long())
                    corr = label[label == output1.to(device)].size(0)
                    correct += corr
                    train_loss.backward()
                    optimizer.step()
                    scheduler.step()
                    tr_total_num += label.size(0)
                    
                train_accuracy = correct/tr_total_num
                train_Loss.append(train_loss.item())
                train_Acc.append(train_accuracy)
                
                corr = 0
                correct = 0
                val_total_num = 0
                best_misclass = 1
                with torch.no_grad():
                    for i,data1 in enumerate(te_loader):
                        x1, label1 = data1
                        x1 = x1.to(device)
                        label1 = label1.squeeze_()
                        label1 = label1.to(device)
                        val_output, b = model(x1)
                        output2 = val_output.argmax(dim=1)
                        val_loss = F.nll_loss(val_output, label1.long())
                        corr = label1[label1 == output2.to(device)].size(0)
                        correct += corr
                        val_total_num += label1.size(0)
                        
                    val_accuracy = correct/val_total_num
                    val_Loss.append(val_loss.item())
                    val_Acc.append(val_accuracy)
                    
                    if (1-val_accuracy) <= best_misclass:
                        best_model = model
                        best_misclass = (1-val_accuracy)
                        
            torch.save(best_model.state_dict(), 'D:/Project/2021/DL/Brain_Switch/Data/Best_model/10_fold/model_{}_{}_{}_{}.pt'.format(temp_subNum,shift+1,fold,num_epoch))
            
            corr=0
            correct=0
            te_total_num=0
            del model, best_model
            model = CNN().to(device)
            model.load_state_dict(torch.load('D:/Project/2021/DL/Brain_Switch/Data/Best_model/10_fold/model_{}_{}_{}_{}.pt'.format(temp_subNum,shift+1,fold,num_epoch)))
            with torch.no_grad():
                for i,data in enumerate(trn_loader):
                    r_x, r_y = data
                    r_x = r_x.to(device)
                    r_y = r_y.squeeze_()
                    r_y = r_y.to(device)
                    train_output, c = model(r_x)
                    
                for i,data in enumerate(te_loader):
                    e_x, e_y = data
                    e_x = e_x.to(device)
                    e_y = e_y.squeeze_()
                    e_y = e_y.to(device)
                    e_output, d = model(e_x)
                    output2 = e_output.argmax(dim=1)
                    corr = e_y[e_y == output2.to(device)].size(0)
                    correct += corr
                    te_total_num += e_y.size(0)
                    
                test_accuracy = correct/te_total_num
                
                te_output = c.cpu().numpy()
                tr_output = d.data.cpu().numpy()
                
                savedict ={'te_data' : te_output,
                           'tr_data' : tr_output}
                
                save_str = 'D:/Project/2021/DL/Brain_Switch/Data/Shallow/10fold/data_{}_{}_{}.mat'.format(temp_subNum,shift+1,fold)
                io.savemat(save_str, savedict)
            del model
            Fold_Acc[fold-1] = test_accuracy
        Shift_Acc[shift] = np.mean(Fold_Acc)
    Final_Acc[temp_subNum - 1] = np.mean(Shift_Acc)
Mean_acc = np.mean(Final_Acc)
Std_acc = np.std(Final_Acc)