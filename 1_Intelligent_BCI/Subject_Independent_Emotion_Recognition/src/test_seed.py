import numpy as np
import torch
import torch.nn as nn
import scipy
import network
from seedloader import *
import torch.utils.data as data
import torch.optim as optim
from torch.autograd import Variable
import pdb
from scipy.stats import pearsonr
import argparse

batch_size = 64
avg_test = 0.

for n in range(15):
	net = network.cnn5d_indep().cuda()
	net.load_state_dict(torch.load('../sbj_indep_model/day1_val{}.pth'.format(str(n))))
	testset = data.DataLoader(dataloader(d_type='test'), batch_size=batch_size)

	net.eval()	
	with torch.no_grad():
		corr = 0.
		tot = 0.
		for i, datas in enumerate(testset):
			x, y, s = Variable(datas[0].cuda()),Variable(datas[1].cuda()), Variable(datas[2].cuda())
			out = net(x)
			_, pred = torch.max(out.data, 1)
			tot += y.size(0)
			corr += (pred == y).sum().item()
		testacc = corr/tot*100
		print('{}th_result: {}'.format(n,testacc))
	avg_test += testacc
  
print('average result: {}'.format(avg_test/15.))




