import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.models as models
import pdb
import torch.backends.cudnn as cudnn
from torch.autograd import Function

class dnn(nn.Module):
    def __init__(self):
        super(dnn, self).__init__()
        
        hdim = 128
        self.custom_net = nn.Sequential()
        
        self.custom_net.add_module('fc1', nn.Linear(85, hdim))
        self.custom_net.add_module('bn1', nn.BatchNorm1d(hdim))
        self.custom_net.add_module('dr1', nn.Dropout2d(0.4))
        self.custom_net.add_module('re1', nn.ELU(True))
        self.custom_net.add_module('fc2', nn.Linear(hdim, hdim))
        self.custom_net.add_module('bn2', nn.BatchNorm1d(hdim))
        self.custom_net.add_module('dr2', nn.Dropout2d(0.4))
        self.custom_net.add_module('re2', nn.ELU(True))
        self.custom_net.add_module('fc3', nn.Linear(hdim, 1))

    def forward(self, x):
        y = self.custom_net(x)
        return y




