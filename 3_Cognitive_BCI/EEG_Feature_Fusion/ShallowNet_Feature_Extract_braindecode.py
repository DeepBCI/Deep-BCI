# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 16:57:16 2022

@author: LJS
"""

### Initial setting ###
import time
import torch
import warnings
from random import shuffle
import numpy as np
import pickle
import mat73
import os
from scipy import io
from braindecode.datautil.signal_target import SignalAndTarget
from braindecode.datautil.splitters import split_into_two_sets
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split

warnings.simplefilter('ignore')

# GPU 사용 설정 (code 수정 필요없이 그대로 쓰면 됨)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
use_cuda = torch.cuda.is_available()

torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

############## 빈칸 채우기! ##############
# Raw data path 설정
path = 'D:/Tak/석박사 9학기/뇌신경신호분석기법/Data' #빈칸#

############## 빈칸 채우기! ##############
# Accuracy 저장을 위한 초기 배열 설정
Final_Acc = np.zeros(1)#빈칸#
sub_list = range(0,2)

for sub in range(1, len(sub_list)):
    print('=============== {} ==============='.format(sub_list[sub]))
    os.chdir(path)
    ############## 빈칸 채우기! ##############
    # Raw data load
    mat_file = mat73.loadmat(path+'/Subject_{}.mat'.format(sub))#빈칸#
    X = mat_file["XX"]
    Y = mat_file["Y"]
    CV_Acc = np.zeros([5])

    # data type 변경
    X = X.astype(np.float32)
    Y = Y.astype(np.int64)  
    
    CV_cnt = 0
    
    # 5-fold 나누는 함수 (아래 함수를 통해 random하게 train, test set이 나뉨, if 5x5-fold, splits = 5, repeats = 5)
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=20220613)
    
    # 5Fold Cross-validation
    for train_index, test_index in rskf.split(X, Y):
        ts = time.time()        # 모델이 각 fold를 training하는데 걸리는 시간을 구하기 위한 함수
        print('( {} )'.format(CV_cnt))
        
        ############## 빈칸 채우기! ##############
        # Raw data - train, test, valid set으로 분할
        X_train, X_test = X[train_index,:,:], X[test_index,:,:]#빈칸#
        y_train, y_test = Y[train_index], Y[test_index]#빈칸#
        XX = len(X_train)
        ind_list = [i for i in range(XX)]
        shuffle(ind_list)
        X_train = X_train[ind_list]#빈칸#
        y_train = y_train[ind_list]#빈칸#
        
        # braindecode 함수를 통해 X, Y를 하나의 set으로 묶기
        train_set = SignalAndTarget(X_train, y_train)
        test_set = SignalAndTarget(X_test, y_test)
        
        # Train set data를 braindecode 함수를 사용하여 train / valid set 나누기 (0.8 비율로)
        train_set, valid_set = split_into_two_sets(train_set, first_set_fraction=0.8)
        del X_train, y_train
        X_train = train_set.X
        X_val = valid_set.X
        y_train = train_set.y
        y_val = valid_set.y
        
        ###  Create the model ##############################################
        # Model load
        from braindecode.models.shallow_fbcsp import ShallowFBCSPNet

        cuda = True
        
        n_classes = 2                           # 분류하려는 class 수
        in_chans = train_set.X.shape[1]         # 사용하려는 data의 channel 수
        
        # https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.23730)
        model = ShallowFBCSPNet(in_chans=in_chans, n_classes=n_classes,
                                input_time_length=None,
                                final_conv_length=20)
        
        if cuda:
            model.cuda()
            
        ### Create cropped iterator #######################################
        # Optimizer 설정 (이 분석에서는 AdamW라는 optimizer 사용)
        from braindecode.torch_ext.optimizers import AdamW
        import torch.nn.functional as F
        
        
        optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.5*0.001)
        model.compile(loss=F.nll_loss, optimizer=optimizer,  iterator_seed=1, cropped=True)
        model.fit(train_set.X, train_set.y, epochs=40, batch_size=16, scheduler='cosine',
                  input_time_length=2000, remember_best_column='valid_misclass',
                  validation_data=(valid_set.X, valid_set.y))  
        
        ### Evaluation ####################################################
        results = model.evaluate(test_set.X, test_set.y)
        
        te = time.time()        # 모델이 각 fold를 training하는데 걸리는 시간을 구하기 위한 함수 (위에 썻던거랑 세트)
        print('time:',round(te-ts,2),'s ''//test loss, test acc :  ' + str(results))
        
        dat_weights = model.network[1].weight.cpu().detach().numpy()
        dat_bias = model.network[1].bias.cpu().detach().numpy()
        out_tr_temporal = model.network[1](torch.tensor(X_train[:,np.newaxis,:,:]).permute(0,1,3,2).cuda()).cpu().detach().numpy()
        out_val_temporal = model.network[1](torch.tensor(X_val[:,np.newaxis,:,:]).permute(0,1,3,2).cuda()).cpu().detach().numpy()
        out_te_temporal = model.network[1](torch.tensor(X_test[:,np.newaxis,:,:]).permute(0,1,3,2).cuda()).cpu().detach().numpy()
        
        tr_Acc = 1-model.epochs_df['train_misclass']
        k = 1-model.epochs_df.train_misclass
        tr_Acc = 1-model.epochs_df.train_misclass[k.shape[0]-1]
        te_Acc = 1-results['misclass']
        CV_Acc[CV_cnt] = 1-results['misclass']
        
        savedict = {'X_train' : X_train,
                    'X_val' : X_val,
                    'X_test' : X_test,
                    'Y_train' : y_train,
                    'Y_val' : y_val,
                    'Y_test' : y_test,
                    'dat_weights' : dat_weights,
                    'dat_bias' : dat_bias,
                    'out_tr_temporal' : out_tr_temporal,
                    'out_val_temporal' : out_val_temporal,
                    'out_te_temporal' : out_te_temporal,
                    'tr_Acc' : tr_Acc,
                    'te_Acc' : te_Acc
                    }
        
        save_str = 'D:/Result/dat_sub_{}_CV_{}_Acc_{}_ver2.mat'.format(sub,CV_cnt,CV_Acc[CV_cnt])
        io.savemat(save_str, savedict)        
        
        CV_cnt = CV_cnt + 1
        del results, optimizer, model, in_chans, n_classes, X_train, X_test, y_train, y_test, train_set, valid_set, test_set
   
    Final_Acc[sub-1] = 1 - sum(CV_Acc) / len(CV_Acc)
