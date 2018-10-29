import torch.utils.data as data
import numpy as np
import torch
from torchvision import transforms

class eegloader(data.Dataset):
	def __init__(self, data_path, split_path, dtype='train', data_dir='./', split_no=0, dlen=160, stpt=320, nch=128):

		data = torch.load(data_dir + data_path)
		split = torch.load(data_dir + split_path)

		self.mean = data['means']
		self.stdev = data['stddevs']
		self.labels = split['splits'][split_no][dtype]

		self.data = []
		for l in self.labels:
			self.data.append(data['dataset'][l])

		assert len(self.data)==len(self.labels)
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

		return x, y, s



