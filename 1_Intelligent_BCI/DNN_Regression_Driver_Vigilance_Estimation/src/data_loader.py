from torch.utils import data
from scipy import io
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from torchvision import datasets
from PIL import Image
import torch
import os
import random
import pdb
import numpy as np


class seedvig(data.Dataset):
    """Dataset class for the seedvig dataset."""

    def __init__(self, data_dir, mode, test_no, data_type):
        """Initialize and preprocess the seedvig dataset."""
        self.data_dir = data_dir
        
        self.data_type = data_type
        self.test_no = test_no
        self.mode = mode
        self.train_dataset = np.empty(0)
        self.train_label = np.empty(0)
        self.val_dataset = np.empty(0)
        self.val_label = np.empty(0)
        self.preprocess()


        if mode == 'train':
            self.num_eeg = self.train_dataset.shape[0]
        else:
            self.num_eeg = self.val_dataset.shape[0]


    def preprocess(self):
        """Preprocess seedvig dataset"""

        data_list = open('{}/data.txt'.format(self.data_dir))
        data_list = data_list.readlines()

        for i in range(23):
            eeg_data = io.loadmat('{}/EEG_Feature_5Bands/{}'.format(self.data_dir, data_list[i][:-1]))[self.data_type].swapaxes(0,1)
            eeg_data = eeg_data.reshape((eeg_data.shape[0],-1))
            eeg_label = io.loadmat('{}/perclos_labels/{}'.format(self.data_dir, data_list[i][:-1]))['perclos']



            n_data = eeg_data.shape[0]
            n_seg = int(n_data/5)
            eeg_tr = np.concatenate((eeg_data[:self.test_no*n_seg],eeg_data[(self.test_no+1)*n_seg:])).copy()
            eeg_try = np.concatenate((eeg_label[:self.test_no*n_seg],eeg_label[(self.test_no+1)*n_seg:])).copy()
            eeg_te = eeg_data[self.test_no*n_seg:(self.test_no+1)*n_seg].copy()
            eeg_tey = eeg_label[self.test_no*n_seg:(self.test_no+1)*n_seg].copy()

            
            if i==0:
                self.train_dataset = eeg_tr
                self.train_label = eeg_try
                self.val_dataset = eeg_te
                self.val_label = eeg_tey
            else:
                self.train_dataset = np.concatenate((self.train_dataset, eeg_tr))
                self.train_label = np.concatenate((self.train_label, eeg_try))
                self.val_dataset = np.concatenate((self.val_dataset, eeg_te))
                self.val_label = np.concatenate((self.val_label, eeg_tey))

        



    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        if self.mode=='train':
            return torch.from_numpy(self.train_dataset[index]).float(), torch.from_numpy(self.train_label[index]).float()
        else:
            return torch.from_numpy(self.val_dataset[index]).float(), torch.from_numpy(self.val_label[index]).float()


    def __len__(self):
        """Return the number of images."""
        return self.num_eeg


def seedvig_loader(eeg_dir, batch_size=128, mode='train', test_no=0, data_type='de_LDS'):
    """Build and return a data loader."""
    dataset = seedvig(eeg_dir, mode, test_no, data_type)
    loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=(mode=='train'), num_workers=1)

    
    return loader
