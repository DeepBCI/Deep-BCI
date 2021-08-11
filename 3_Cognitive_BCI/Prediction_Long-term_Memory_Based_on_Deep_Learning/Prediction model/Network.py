import torch.nn as nn
import torch.nn.functional as F

# CNN model
class CNN(nn.Module):
    def __init__(self, outputSize):
        self.outputSize=outputSize
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(6, 8, kernel_size=3)  # Training: inputs.size(271,6,10,9) / Testing: inputs.size(31,6,10,9)
        self.bn = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 64, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(64)

        self.mp = nn.MaxPool2d(2)

        self.do = nn.Dropout(0.3)

        self.fc = nn.Linear(384, self.outputSize)

    def forward(self, x):
        x = F.relu(self.bn(self.conv1(x)))
        # print(x.size())
        x = F.relu(self.bn2(self.conv2(x)))
        # print(x.size())
        x = self.mp(x)
        # print(x.size())
        x = x.view(x.shape[0], -1)  # linear input size
        x = self.do(x)
        # print(x.size())
        x = self.fc(x)
        # print(x.size())

        return x

class CNN2(nn.Module):
    def __init__(self, outputSize):
        self.outputSize=outputSize
        super(CNN2, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(6, 6, kernel_size=3, groups=6),  # Training: inputs.size(271,6,10,9) / Testing: inputs.size(31,6,10,9)
            nn.Conv2d(6, 16, kernel_size=1))
        self.bn = nn.BatchNorm2d(16)
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, groups=16),
            nn.Conv2d(16, 64, kernel_size=1))
        self.bn2 = nn.BatchNorm2d(64)

        self.mp = nn.MaxPool2d(2)

        self.do = nn.Dropout(0.3)

        self.fc = nn.Linear(384, self.outputSize)



    def forward(self, x):
        x = F.relu(self.bn(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.mp(x)
        x = x.view(x.shape[0], -1)  # linear input size
        x = self.do(x)
        x = self.fc(x)

        return x


class MLP(nn.Module):
    def __init__(self, outputSize):
        super(MLP, self).__init__()

        self.outputSize = outputSize
        self.linear1 = nn.Linear(360, 180)
        self.linear2 = nn.Linear(180, 90)
        self.linear3 = nn.Linear(90, self.outputSize)
        self.do = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        # print(x.size())
        x = F.relu(self.linear2(x))
        # print(x.size())
        x = self.do(x)
        # print(x.size())
        x = self.linear3(x)
        # print(x.size())

        return x


