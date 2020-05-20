import torch.utils.data as data
import numpy as np
from dataloader import *
from model import *
import torch.nn
import torch.optim as optim
from torch.autograd import Variable
import torch

try:
	torch._utils._rebuild_tensor_v2
except AttributeError:
	def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
		tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
		tensor.requires_grad = requires_grad
		tensor._backward_hooks = backward_hooks
		return tensor
	torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2


data_dir = '../data/'
batch_size = 128
learning_rate = 0.001
model_path = '../model/'
use_saved_model = False

hidden_dim = 128
num_class = 40
data_path = 'eeg_signals_128_sequential_band_all_with_mean_std.pth'
split_path = 'splits_by_image.pth'
split_no = 3

model_name_to_save = 'GRU_h128_b128.pt'
saved_model = model_name_to_save

trainset = data.DataLoader(eegloader(data_path, split_path, dtype='train', data_dir=data_dir, split_no=split_no),batch_size=batch_size, shuffle=True)
valset = data.DataLoader(eegloader(data_path, split_path, dtype='val', data_dir=data_dir, split_no=split_no),batch_size=batch_size)
testset = data.DataLoader(eegloader(data_path, split_path, dtype='test', data_dir=data_dir, split_no=split_no),batch_size=batch_size)
print('data loaded')


net = RNN_Encoder(num_class, hidden_dim=hidden_dim).cuda()
criterion = torch.nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.Adam(net.parameters(), lr = learning_rate)
print(net)

if use_saved_model:
	net.load_state_dict(torch.load(model_path+saved_model))
	print('saved model loaded')

epoch = 200
prev_loss = 9999
prev_acc = 0.

print('start train/test')
for n in range(epoch):

	net.train()

	ntr = 0.
	nva = 0.
	nte = 0.
	
	loss_tr = 0.
	loss_va = 0.
	loss_te = 0.

	corr_tr = 0.
	corr_va = 0.
	corr_te = 0.
	
	for i, data in enumerate(trainset):
		x_tr,y_tr,yid = Variable(data[0].cuda()),Variable(data[1].cuda()),data[2]
		optimizer.zero_grad()
		out = net(x_tr)
		loss = criterion(out, y_tr)
		loss.backward()
		optimizer.step()
		loss_tr+= float(loss.data[0])
		_, pred = torch.max(out, 1)
		corr_tr += float((pred==y_tr).sum())
		ntr += float(y_tr.size(0))

	net.eval()
	
	with torch.no_grad():
		for i, data in enumerate(valset):
			x_va,y_va,yid_va = Variable(data[0].cuda()),Variable(data[1].cuda()),data[2]
			out = net(x_va)
			loss = criterion(out,y_va)
			loss_va+= float(loss.data[0])
			_, pred = torch.max(out,1)
			corr_va += float((pred==y_va).sum())
			nva += float(y_va.size(0))

		for i, data in enumerate(testset):
			x_te,y_te,yid_te = Variable(data[0].cuda()),Variable(data[1].cuda()),data[2]
			out = net(x_te)
			loss = criterion(out,y_te)
			loss_te+= float(loss.data[0])
			_, pred = torch.max(out,1)
			corr_te += float((pred==y_te).sum())
			nte += float(y_te.size(0))

		if prev_acc < corr_va/nva:
			prev_acc = corr_va/nva
			torch.save(net.state_dict(),model_path+model_name_to_save)
			print('> Max validation accuracy: {:.4f} model saved'.format(prev_acc))

		print('{:04d} epoch train loss: {:.4f} test loss: {:.4f}'.format(n+1,float(loss_tr/ntr),float(loss_te/nte)))
		print('           train acc:  {:.4f} val acc: {:.4f} test acc: {:.4f}' .format(corr_tr/ntr, corr_va/nva, corr_te/nte))

print('finished')
