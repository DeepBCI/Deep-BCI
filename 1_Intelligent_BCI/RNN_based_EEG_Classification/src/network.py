import torch.nn as nn


class RNN_Encoder(nn.Module):
	def __init__(self, num_cls=6, model_type='GRU', hidden_dim=70, input_dim=70):
		super(RNN_Encoder, self).__init__()
		self.hidden_dim = hidden_dim 
		self.input_dim = input_dim
		self.output_dim = num_cls
	
		if model_type=='GRU':
			self.rnn = nn.GRU(self.input_dim, self.hidden_dim, num_layers = 1,  batch_first = True)
		elif model_type=='LSTM':
			self.rnn = nn.LSTM(self.input_dim, self.hidden_dim, num_layers = 1, batch_first=True) 
		self.fc1 = nn.Linear(self.hidden_dim, self.hidden_dim)
		self.fc2 = nn.Linear(self.hidden_dim, self.output_dim)

	def forward(self, feature, get_feat=False):

		feature = feature.transpose(1,2) #dimension change : batch x time x dimension	
		x, hidden = self.rnn(feature)
		x = x.select(1, x.size(1)-1).contiguous()
		x = x.view(-1, self.hidden_dim)
		x = self.fc1(x)
		result = self.fc2(x)

		if get_feat:
			return x
		return result
