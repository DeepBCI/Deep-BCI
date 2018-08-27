from network import RNN_Encoder
import torch.nn
import torch.utils.data as data
from dataloader import *
import torch.optim as optim
from torch.autograd import Variable


data_dir = '../data/'
data_path = 'dlist.txt' # data list
num_class = 6
learning_rate = 0.001
model_path = '../model/' # path to save model
model_name_to_save = 'eeg_6class_simple_gru.pth' # name of model to save
use_saved_model = False
batch_size = 32
nEpoch = 100
prev_acc = 0

trainset = data.DataLoader(dataloader(data_path, dtype='train'), batch_size=batch_size, shuffle=True)
valset = data.DataLoader(dataloader(data_path, dtype='test'), batch_size=batch_size)

net = RNN_Encoder(num_class, model_type='GRU', hidden_dim=70).cuda()
print(net)
criterion = torch.nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

if use_saved_model:
	print('saved model loaded')
	net.load_state_dict(torch.load(model_path+model_name_to_save))

print('start train/test')

for n in range(nEpoch):
	net.train()

	ntr = 0.
	loss_tr = 0.
	corr_tr = 0.
	# train	
	for i, data in enumerate(trainset):
		x_tr, y_tr = Variable(data[0].cuda()), Variable(data[1].cuda())
		optimizer.zero_grad()
		out = net(x_tr)
		loss = criterion(out, y_tr)
		loss.backward()
		optimizer.step()
		loss_tr += float(loss.data[0])
		_, pred = torch.max(out, 1)
		corr_tr += float((pred==y_tr).sum())
		ntr += float(y_tr.size(0))

	nte = 0.
	loss_te = 0.
	corr_te = 0.
	# test
	net.eval()	
	for i, data in enumerate(valset):
		x_te, y_te = Variable(data[0].cuda()), Variable(data[1].cuda())
		out = net(x_te)
		loss = criterion(out, y_te)
		_, pred = torch.max(out, 1)
		loss_te += float(loss.data[0])
		corr_te += float((pred==y_te).sum())
		nte += float(y_te.size(0))
		
	print('{:03d} epoch train acc: {:.4f} train loss: {:.4f}'.format(n+1, corr_tr/ntr, float(loss_tr)))
	print('          val acc: {:.4f}   val loss: {:.4f}'.format(corr_te/nte, float(loss_te)))

	if prev_acc < corr_te/nte:
		prev_acc = corr_te/nte
		torch.save(net.state_dict(), model_path+model_name_to_save)
		print('model saved')
