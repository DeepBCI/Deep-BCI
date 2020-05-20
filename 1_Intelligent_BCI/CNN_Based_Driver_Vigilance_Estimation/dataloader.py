from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from torchvision import datasets
from PIL import Image
import torch
import os
import random
import pdb
import numpy as np

class dataloader(data.Dataset):
	
	def __init__(self, data_name='1_20151124_noon_2.npy', d_type='train', data_dir = '/mnt/dataset/eeg/seedvig/DE_LDS_1/'):
		if d_type == 'train':
			dx = 'tr_' # training eeg
			dy = 'try_' # training eeg labels
		else:
			dx = 'te_' # test eeg
			dy = 'tey_' # test eeg labels

		self.x = torch.from_numpy(np.load(data_dir+dx+data_name)).float()
		self.y = torch.from_numpy(np.load(data_dir+dy+data_name)).float()

		self.l = int(self.x.shape[0]) # number of samples (train/test)


	def __len__(self):
		return self.l
	
	def __getitem__(self, idx):

		return self.x[idx], self.y[idx]
