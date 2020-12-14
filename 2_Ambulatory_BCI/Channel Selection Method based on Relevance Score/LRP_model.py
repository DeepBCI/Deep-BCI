

from __future__ import print_function

import torch
import torch.nn as nn

import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        

        self.conv1 = nn.Conv2d(118, 50, kernel_size=(20,9), stride= (1,1), padding =(1,1) )
        self.conv2 = nn.Conv2d(50, 80, kernel_size=(10,5), stride= (1,1), padding =(1,1) )
        self.conv3 = nn.Conv2d(80, 100, kernel_size=(6,3), stride= (1,1), padding =(1,1) )

        
        self.max_pool1 = nn.MaxPool2d(kernel_size =(3,2))
        self.max_pool2 = nn.MaxPool2d(kernel_size =(3,2))
        self.max_pool3 = nn.MaxPool2d(kernel_size =(3,2))

        self.conv3_drop = nn.Dropout2d()


        self.softmax = nn.LogSoftmax(dim=1)
        self.relu = nn.ReLU()
        
      
        self.fc1 = nn.Linear(2500, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 64)
        self.fc4 = nn.Linear(64, 2)

 


    def forward(self, x):


        x = self.relu(self.max_pool1(self.conv1(x)))

        x = self.relu(self.max_pool2(self.conv2(x)))

        x = self.relu(self.max_pool3(self.conv3_drop(self.conv3(x))))
        #x = self.relu(self.max_pool3(self.conv3(x)))


        x = x.view(-1, 2500)
        x = self.relu(self.fc1(x))
        x = self.conv3_drop(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)


        return self.softmax(x)



def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):

       
        target=target.long()
        optimizer.zero_grad()
        output = model(data)
        #print("output", output)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


