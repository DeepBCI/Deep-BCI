import numpy as np
import torch
import torch.utils.data as data

class dataloader(data.Dataset):
	def __init__(self, data_path, dtype='train', data_dir='../data/'):
		all_data = open(data_dir+data_path).readlines()
		all_data = [elm[:-1] for elm in all_data]

		self.data = []
		self.y = []
		self.ndata = 0
		self.i = 0

		for j in range(len(all_data)):
			data = np.load(data_dir+all_data[j])
			if dtype=='train': # 501 to 1500 (for train)
				data = data[501:1501,1:71] # [time, dimension(channelxfrequency band)]
				self.ndata = 50
				self.i = 0
			elif dtype=='test': # 1501 to 2000 (for test)
				data = data[1501:2001,1:71]
				self.ndata = 20
				self.i = 1501
			data = data.astype(float)
			for i in range(self.ndata):
				data_for_append = data[(i*20):(i+1)*20,:]
				data_for_append.shape = 70, -1
				self.data.append(data_for_append)
				self.y.append(j)
	
	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		x = torch.from_numpy(self.data[idx]).type(torch.FloatTensor)
		y = self.y[idx]
		return x, y
		
