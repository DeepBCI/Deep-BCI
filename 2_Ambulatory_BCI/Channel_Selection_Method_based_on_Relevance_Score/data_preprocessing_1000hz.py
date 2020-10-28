
# declare packages
import torch
import numpy as np
import scipy.io as scio
import matplotlib as plt

from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader 
import hdf5storage as hdf


# declare data_class
class EEG_train_dataset(Dataset):
    '''EEG dataset'''

    # data load, preprocessing
    def __init__(self):
        
        # load kfold_data file
        
        x_train = scio.loadmat('data_1000hz/stft_av/x_train1', squeeze_me=True)
        y_train = scio.loadmat('data_1000hz/stft_av/y_train1', squeeze_me=True)


        # extract signal only
        
        x_train = x_train['xtrain']
        y_train = y_train['ytrain']
        
        
        # make tensor
        x_train = torch.from_numpy(x_train)
        y_train = torch.from_numpy(y_train)

        # x_train.shape = [252,18,200, 200]
        # y_train.shape = [252,1]

        
        # vrnn input shape에 맞게 변형
        
        # .unsqueeze-> 3d version
        #self.train_data = x_train.unsqueeze(1)
        
        self.train_data = x_train
        self.train_label = y_train

        
    def __getitem__(self, index):
        _x = self.train_data[index]
        _y = self.train_label[index]
        return _x, _y

    def __len__(self):
        return len(self.train_data)


class EEG_test_dataset(Dataset):
    '''EEG dataset'''

    # data load, preprocessing
    def __init__(self):       
        # load kfold_data file
        
        
        x_test = scio.loadmat('data_1000hz/stft_av/x_test1', squeeze_me=True)
        y_test = scio.loadmat('data_1000hz/stft_av/y_test1', squeeze_me=True)

        # extract signal only
        #x_test = x_test['freq_reduced_xtest']
        
        x_test = x_test['xtest']
        y_test = y_test['ytest']

        # make tensor
        x_test = torch.from_numpy(x_test).double()
        y_test = torch.from_numpy(y_test).double()


        # x_test.shape = [28,18,258,24]
        # y_test.shape = [28,1]

        # .unsqueeze-> 3d version
        #self.test_data = x_test.unsqueeze(1)
        self.test_data = x_test
        self.test_label= y_test
   
    def __getitem__(self, index):
        _x = self.test_data[index]
        _y = self.test_label[index]
        return _x, _y

    def __len__(self):
        return len(self.test_data)

   

