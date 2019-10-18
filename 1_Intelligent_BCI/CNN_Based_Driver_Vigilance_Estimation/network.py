import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class cnnvig(nn.Module):
  def __init__(self):
    super(cnnvig, self).__init__()
    self.conv1 = nn.Conv2d(5, 6, 5) 
    self.conv2 = nn.Conv2d(6, 16, 5) 
    self.pool = nn.MaxPool2d(2,2)
    self.dropout = nn.Dropout2d()
    self.fc1 = nn.Linear(16*5*5, 120) 
    self.fc2 = nn.Linear(120, 50, bias=False)
    self.fc3 = nn.Linear(50, 1, bias=False)

  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(self.dropout(F.relu(self.conv2(x))))
    x = x.view(-1, 16*5*5)
    x = F.relu(self.fc1(x))
    x_ = self.fc2(x)
    x = F.dropout(F.relu(x_),training=self.training)
    x = self.fc3(x)
    return x







