import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.models as models
import pdb
import torch.backends.cudnn as cudnn
from torch.autograd import Function

torch.manual_seed(1234)

class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class dnn_cls_da(nn.Module):
    def __init__(self, hdim=128, in_dim=60, dr=0.5):
        super(dnn_cls_da, self).__init__()
        
        hdim = hdim
        input_dim = in_dim 

        self.custom_net = nn.Sequential()
        
        self.custom_net.add_module('fc1', nn.Linear(input_dim, hdim))
        self.custom_net.add_module('bn1', nn.BatchNorm1d(hdim))
        self.custom_net.add_module('dr1', nn.Dropout2d(dr))
        self.custom_net.add_module('re1', nn.ELU(True))
        self.custom_net.add_module('fc2', nn.Linear(hdim, hdim))
        self.custom_net.add_module('bn2', nn.BatchNorm1d(hdim))
        self.custom_net.add_module('dr2', nn.Dropout2d(dr))
        self.custom_net.add_module('re2', nn.ELU(True))

        self.cls = nn.Sequential()
        self.cls.add_module('fc3', nn.Linear(hdim, hdim))
        self.cls.add_module('bn3', nn.BatchNorm1d(hdim))
        self.cls.add_module('dr3', nn.Dropout2d(dr))
        self.cls.add_module('re3', nn.ELU(True))
        self.cls.add_module('fc4', nn.Linear(hdim, hdim))
        self.cls.add_module('bn4', nn.BatchNorm1d(hdim))
        self.cls.add_module('dr4', nn.Dropout2d(dr))
        self.cls.add_module('re4', nn.ELU(True))
        self.cls.add_module('fc5', nn.Linear(hdim, 3)) #awake, tired, drowsy

        self.da = nn.Sequential()
        self.da.add_module('fc3_da', nn.Linear(hdim, hdim))
        self.da.add_module('bn3_da', nn.BatchNorm1d(hdim))
        self.da.add_module('dr3_da', nn.Dropout2d(dr))
        self.da.add_module('re3_da', nn.ELU(True))
        self.da.add_module('fc4_da', nn.Linear(hdim, hdim))
        self.da.add_module('bn4_da', nn.BatchNorm1d(hdim))
        self.da.add_module('dr4_da', nn.Dropout2d(dr))
        self.da.add_module('re4_da', nn.ELU(True))
        self.da.add_module('fc5_da', nn.Linear(hdim, 8)) # number of trials 


        self._initialize_weights()

        self.softmax = nn.Softmax(dim=1)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


    def forward(self, x, alpha=0):
        f = self.custom_net(x)
        y = self.cls(f)
        
        f_rev = ReverseLayerF.apply(f, alpha)
        y_d = self.da(f_rev)

        return y, y_d


