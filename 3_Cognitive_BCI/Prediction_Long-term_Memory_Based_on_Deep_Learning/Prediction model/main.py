import os

import torch.utils.data
import numpy as np

import torch.nn as nn
import torch.utils.data
import torch.optim as optim
import matplotlib.pyplot as plt
import DataPreprocessing as DP
from training import training

from Network import MLP, CNN, CNN2

DP.gpu_setting(gpu_num=0)

# hyper-parameter setting
learning_rate = 1e-5
batchSize = 20
start_epoch = 0
num_epoch = 50

# file path
datapath = os.getcwd()

# Picture memory
datatype = './PM_PSD_24H/'
# Location memory
# datatype='./LM_PSD_24H/'

# 1D
x_datapath = 'x_data/'
y_datapath = 'y_data/'

# 2D mesh (only CNN)
# x_datapath = 'x_2Ddata/'
# y_datapath = 'y_2Ddata/'

x_filepath = datapath + datatype + x_datapath
y_filepath = datapath + datatype + y_datapath

x_filelist = os.listdir(x_filepath)
y_filelist = os.listdir(y_filepath)

# sort
x_filelist.sort()
y_filelist.sort()

# file load
x_data = []
y_data = []

for i in range(len(x_filelist)):
    x_data.append(DP.preprocess_data(x_filepath, x_filelist[i]))
    y_data.append(DP.load_header(y_filepath, y_filelist[i]))


acc_mean = []
kappa_mean = []
conf_matrix_kappa_max = []
for fold in range(len(x_data)):

    [trainx, trainy, testx, testy] = DP.split_train_test(x_data, y_data, fold)

    # CNN
    # trainx = np.transpose(trainx, (3, 2, 0, 1))
    # testx = np.transpose(testx, (3, 2, 0, 1))

    # MLP
    trainx = np.transpose(trainx, (1, 0))
    testx = np.transpose(testx, (1, 0))

    trainx = torch.FloatTensor(trainx).cuda()
    testx = torch.FloatTensor(testx).cuda()
    trainy = torch.LongTensor(trainy).cuda()
    testy = torch.LongTensor(testy).cuda()

    train = torch.utils.data.TensorDataset(trainx, trainy)
    test = torch.utils.data.TensorDataset(testx, testy)

    # batch iterator
    trainloader = torch.utils.data.DataLoader(train, batch_size=batchSize, shuffle=True)
    testloader = torch.utils.data.DataLoader(test, batch_size=batchSize, shuffle=False)

    # model define (Selected MLP or CNN)
    net = MLP(2) # outputsize
    net.cuda()

    # loss define (cross-entropy)
    criterion = nn.CrossEntropyLoss()

    # Optimizer define
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, weight_decay=1e-9)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epoch)

    setting = {'net': net,
               'trainloader': trainloader,
               'testloader': testloader,
               'optimizer': optimizer,
               # 'scheduler': scheduler,
               'loss': criterion,
               'max_epoch': num_epoch,
               'fold': fold
               }

    [training_losses, kappa_epoch, test_losses, acc_epoch, conf_matrix_all] = training(setting)

    # loss
    plt.figure()
    plt.plot(training_losses, label='Training loss')
    plt.plot(test_losses, label='Test loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(frameon=False)
    plt.savefig(datatype+'results_MLP/Training'+str(fold)+'.png')
    plt.close()

    # Acc
    plt.figure()
    plt.plot(acc_epoch, label='Val Accuracy/Epochs')
    plt.legend("")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(frameon=False)
    plt.savefig(datatype+'results_MLP/Acc'+str(fold)+'.png')
    plt.close()

    # kappa
    plt.figure()
    plt.plot(kappa_epoch, label='Val Kappa Score/Epochs')
    plt.legend("")
    plt.xlabel("Epochs")
    plt.ylabel("Kappa Score")
    plt.legend(frameon=False)
    plt.savefig(datatype+'results_MLP/Kappa'+str(fold)+'.png')
    plt.close()

    print('[MAX Accuracy] %.3f(%d), [MAX Kappa] %.3f(%d)' % (np.max(acc_epoch), np.argmax(acc_epoch), np.max(kappa_epoch), np.argmax(kappa_epoch)))

    # acc / kappa
    acc_mean.append(np.max(acc_epoch))
    kappa_mean.append(np.max(kappa_epoch))
    # confusion matrix
    conf_matrix_kappa_max.append(conf_matrix_all[np.argmax(kappa_epoch)])

# average accuracy
Acc_avg = np.mean(acc_mean)
Acc_std = np.std(acc_mean)
# kappa accuracy
kappa_avg = np.mean(kappa_mean)
kappa_std = np.std(kappa_mean)

# confusion matrix
conf_max = sum(conf_matrix_kappa_max)
print(conf_max)
print(conf_max.astype('float') / conf_max.sum(1)[:, None])

print('[Average Acc] %.3f(%.3f), [Average Kappa] %.3f(%.3f)' % (Acc_avg, Acc_std, kappa_avg, kappa_std))
