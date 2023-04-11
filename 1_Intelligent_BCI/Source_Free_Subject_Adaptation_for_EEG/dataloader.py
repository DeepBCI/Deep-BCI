import torch.utils.data as data
import numpy as np
import torch
from torchvision import transforms
import random
import utils

class eegloader(data.Dataset):
	def __init__(self, data_path, split_path, dtype='train', split_no=0, dlen=160, stpt=320, nch=128, seed=-1, subject_id=None, target_subject=None, k_shot=50, num_classes=40):
		if seed >= 0:
			utils.set_seed(seed)

		data = torch.load(data_path)
		split = torch.load(split_path)

		self.mean = data['means']
		self.stdev = data['stddevs']
		self.labels = split['splits'][split_no][dtype]

		self.data = []
		class_accum = [0] * num_classes
		for l in self.labels:
			if subject_id is not None:
				if data['dataset'][l]['subject'] != (subject_id + 1):
					continue

			if dtype == 'train':
				if data['dataset'][l]['subject'] == (target_subject + 1):
					if class_accum[data['dataset'][l]['label']] < k_shot:
						self.data.append(data['dataset'][l])
						class_accum[data['dataset'][l]['label']] += 1
				else:
					self.data.append(data['dataset'][l])
			else:
				if data['dataset'][l]['subject'] == (target_subject + 1):
					self.data.append(data['dataset'][l])

		self.dlen = dlen
		self.stpt = stpt
		self.nch = nch

	def __len__(self):
		return len(self.data)
	
	def __getitem__(self, idx):
		nch  = self.nch
		dlen = self.dlen
		stpt = self.stpt
	
		x = np.zeros((nch,dlen))
		y = self.data[idx]['label']
		s = self.data[idx]['subject'] 
		
		x = torch.from_numpy(x)
		x[:,:min(int(self.data[idx]['eeg'].shape[1]),dlen)] = self.data[idx]['eeg'][:,stpt:stpt+dlen]
		x = x.type(torch.FloatTensor).sub(self.mean.expand(nch,dlen))/ self.stdev.expand(nch,dlen)

		return x, y, s-1



