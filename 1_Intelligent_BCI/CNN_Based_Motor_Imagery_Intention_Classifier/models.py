import torch.nn as nn
import torch

class EEGNet(nn.Module):
    def __init__(self,num_classes, input_ch,input_time, batch_norm=True,
                 batch_norm_alpha=0.1):
        super(EEGNet, self).__init__()
        self.batch_norm = batch_norm
        self.batch_norm_alpha = batch_norm_alpha
        self.n_classes = num_classes
        freq = 250
        self.convnet = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(1, freq//2), stride=1, bias=False, padding=(0, freq//4)),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, kernel_size=(input_ch, 1), stride=1, groups=8),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1,4)),
            nn.Dropout(p=0.25),
            nn.Conv2d(16, 16 , kernel_size=(1,freq//4),padding=(0,freq//8), groups=16),
            nn.Conv2d(16, 16, kernel_size=(1,1)),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(p=0.25)
            )
        self.convnet.eval()
        out = self.convnet(torch.zeros(1, 1, input_ch, input_time))
        self.num_hidden = out.size()[1] * out.size()[2] * out.size()[3]
        from pytorch_model_summary import summary
        print(summary(self.convnet, torch.zeros(1, 1, input_ch, input_time), show_input=False))

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)

        return output

class FcClfNet(nn.Module):
    def __init__(self, embedding_net, l2norm=False):
        super(FcClfNet, self).__init__()
        self.embedding_net = embedding_net
        self.num_hidden = embedding_net.num_hidden
        self.clf = nn.Sequential(nn.Linear(embedding_net.num_hidden, embedding_net.n_classes),
                                 nn.Dropout())
        self.l2norm=l2norm
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        output = self.embedding_net(x)
        output = self.clf(output)
        output = output.view(output.size()[0], self.embedding_net.n_classes, -1)
        if output.size()[2]==1:
            output = output.squeeze()

        return output
