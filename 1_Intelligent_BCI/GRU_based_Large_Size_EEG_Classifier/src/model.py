import torch.nn as nn
import torch.nn.functional as F


class RNN_Encoder(nn.Module):
	def __init__(self, num_cls=40, hidden_dim=64, input_dim=128):
		super(RNN_Encoder, self).__init__()
		self.hidden_dim = hidden_dim 
		self.input_dim = input_dim
		self.output_dim = num_cls 
	
		self.rnn = nn.GRU(self.input_dim, self.hidden_dim, num_layers=1,  batch_first=True)
		

	def forward(self, feature, get_feat=False):
		feature = feature.transpose(1,2)	
		x, hidden = self.rnn(feature)
		x = x.select(1, x.size(1)-1).contiguous()
		x = x.view(-1, self.hidden_dim)
		x = F.leaky_relu(self.fc1(x),0.2)
		result = self.fc2(x)

		if get_feat:
			return x
		return result
