
from __future__ import print_function

# declare packages
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
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def val(args, model, device, val_loader):
    del args
    model.eval()
    val_loss = 0
    val_correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            target = target.long()
            output = model(data)
            #print("output",output)
            val_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            val_correct += pred.eq(target.view_as(pred)).sum().item()


    val_loss /= len(val_loader.dataset)
    print('\n Val set: Average loss: {:.4f}, Val_accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, val_correct, len(val_loader.dataset),
        100. * val_correct / len(val_loader.dataset)))
    val_accuracy = 100. * val_correct / len(val_loader.dataset)
    
    return val_loss, val_accuracy



def test(args, model, device, test_loader):
    del args
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            target = target.long()
            output = model(data)
            #print("output",output)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()


    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    test_accuracy = 100. * correct / len(test_loader.dataset)
    
    return test_accuracy
