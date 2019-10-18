import numpy as np
import torch
import torch.nn as nn
import scipy
import network
from dataloader import *
import torch.utils.data as data
import torch.optim as optim
from torch.autograd import Variable
import pdb
from scipy.stats import pearsonr
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('num', type=int)
parser.add_argument('splits', type=int)
args = parser.parse_args()

lists = open('data list').readlines()
lists = [elm.split('.')[0] for elm in lists]
splits = 'data split path'
batch_size = 64
correlations = 0.
testloss = 0.
 
n_epoch = 10
for sbj in range(21):
  for splno in range(1,6):
    train_data = lists[sbj] + '.npy'
    splits = '/mnt/dataset/eeg/seedvig/PSD_LDS_'+str(splno)+'/'
    net = network.cnnvig().cuda()
    trainset = data.DataLoader(dataloader(train_data, d_type='train', data_dir=splits), batch_size=batch_size, shuffle=True)
    testset = data.DataLoader(dataloader(train_data, d_type='test', data_dir=splits), batch_size=177)
    
    criterion = torch.nn.MSELoss(size_average=False).cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr = 0.0001)
    save_name = '/mnt/EEG/SEED-VIG/model/psd_lds_'+lists[sbj]+'_'+str(splno)+'.pth'
       
    for ep in range(n_epoch):
    	net.train()
    
    	loss_tr = 0.
    	for i, datas in enumerate(trainset):
    		x, y = Variable(datas[0].cuda()),Variable(datas[1].cuda())
    		optimizer.zero_grad()
    		out = net(x)
    		loss = criterion(out, y)
    		loss.backward()
    		optimizer.step()
    		loss_tr += float(loss.item())
    
    	net.eval()
    
    
    	with torch.no_grad():
    		loss_te = 0.
    		corr_te = []
    		for i, datas in enumerate(testset):
    			x, y = Variable(datas[0].cuda()),Variable(datas[1].cuda())
    			out = net(x)
    			loss = criterion(out, y)
    			loss_te += float(loss.item())
    			out_v = out.cpu().data.numpy()
    			out_v.shape = -1
    			y_v = y.cpu().data.numpy()
    			y_v.shape = -1
    			cc, pp= pearsonr(out_v, y_v)
    			corr_te.append(float(cc))
    
    torch.save(net.state_dict(), save_name)
    print('{} {} train loss = {:.04f}, test loss = {:.04f} correlation = {:.04f}'.format(lists[sbj],splno,loss_tr/708., loss_te/177., corr_te[0]))
    if corr_te[0]>0:
      correlations += float(corr_te[0])
    testloss += float(loss_te/177)

  

print(correlations/5., testloss/5.)




