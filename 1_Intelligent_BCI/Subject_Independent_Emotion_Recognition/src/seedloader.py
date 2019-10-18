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
	def __init__(self, dlist='../day1.txt', d_type='train', valno=0, data_dir = '../'):

		f = open(dlist)
		name = f.readlines()
		name = [elm[:-1] for elm in name]
		f.close()

		x,y,s = [], [], []

		if d_type=='train' or d_type=='test':	
			for i in range(15):
				if i==valno:
					continue
				x_name = '{}/{}_de_lds_x.npy'.format(d_type, name[i])
				y_name = '{}/{}_de_lds_y.npy'.format(d_type, name[i])
	
				self.x = torch.from_numpy(np.load(data_dir+x_name)).float()
				self.y = torch.from_numpy(np.load(data_dir+y_name)).long()
				self.s = self.x.shape[0]*[i]
				x.append(self.x)
				y.append(self.y)
				s.append(self.s)
			self.x = np.concatenate(x)
			self.y = np.concatenate(y)
			self.s = np.concatenate(s)
			self.l = self.x.shape[0]

		else:
			i = valno
			x_name = 'train/{}_de_lds_x.npy'.format(name[i])
			y_name = 'train/{}_de_lds_y.npy'.format(name[i])
			self.x = torch.from_numpy(np.load(data_dir+x_name)).float() 
			self.y = torch.from_numpy(np.load(data_dir+y_name)).long()
			self.s = np.array([i]*self.x.shape[0])
			self.l = self.x.shape[0]
	

	def __len__(self):
		return self.l
	
	def __getitem__(self, idx):

		return self.x[idx], self.y[idx], self.s[idx]
