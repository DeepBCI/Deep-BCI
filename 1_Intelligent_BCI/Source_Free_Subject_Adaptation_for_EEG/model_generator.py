import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
	def __init__(self, input_dim=100, output_dim=64, hidden_dim=64, num_classes=40):
		super(Generator, self).__init__()
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.hidden_dim = hidden_dim
		self.num_classes = num_classes

		self.label_emb = nn.Embedding(self.num_classes, self.input_dim)

		self.fc = nn.Sequential(
			nn.Linear(self.input_dim, self.hidden_dim),
			nn.ReLU(),
			nn.Linear(self.hidden_dim, self.hidden_dim)
		)
			
	def forward(self, z, label):
		x = torch.mul(self.label_emb(label), z)
		x = self.fc(x)
		return x