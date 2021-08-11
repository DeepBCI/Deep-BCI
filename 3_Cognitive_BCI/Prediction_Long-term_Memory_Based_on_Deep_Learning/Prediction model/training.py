import numpy as np

import torch
import torch.utils.data

from sklearn.metrics import cohen_kappa_score, accuracy_score
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

def testing(epoch, net, testloader, criterion):

    # global bestAcc, inputs, aa
    net.eval()
    testLoss = 0

    batchSize = testloader.batch_size
    with torch.no_grad():
        y_act_all = np.zeros(testloader.dataset.tensors[0].shape[0])
        y_pred_all = np.zeros(testloader.dataset.tensors[0].shape[0])
        for batchIdx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.view(-1).cuda()

            outputs = net(inputs)
            # calculate the batch loss
            loss = criterion(outputs, targets)
            # update average test loss
            testLoss += loss.item()

            # cohen_kappa_score_kappa_score
            y_actual = targets.data.cpu().numpy()
            y_pred = outputs.detach().cpu().numpy()
            y_pred = np.argmax(y_pred, axis=1)
            y_act_all[batchIdx*batchSize:(batchIdx+1)*batchSize] = y_actual
            y_pred_all[batchIdx*batchSize:(batchIdx+1)*batchSize] = y_pred

            # confusion matrix
            # auc = roc_auc_score(y_actual, y_pred)
            # print('ROC AUC: %f' % auc)

        testLoss = testLoss/len(testloader.sampler)

        # kappa
        testing_kappa = cohen_kappa_score(y_act_all, y_pred_all)
        # accuracy
        testing_acc = 100 * (accuracy_score(y_act_all, y_pred_all))
        # confusion matrix
        conf_matrix = confusion_matrix(y_act_all, y_pred_all)

    return (testing_kappa, testLoss, testing_acc, conf_matrix)

def training(setting):
    net = setting['net']
    trainloader = setting['trainloader']
    testloader = setting['testloader']
    optimizer = setting['optimizer']
    # scheduler = setting['scheduler']
    criterion = setting['loss']
    max_epoch = setting['max_epoch']
    fold = setting['fold']

    net.train()
    training_losses = []
    acc_epoch = []
    kappa_epoch = []
    test_losses = []
    conf_matrix_all = []
    for i in tqdm(range(max_epoch), desc='Training '+str(fold+1)):
        trainLoss = 0

        for batchIdx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.cuda(), targets.view(-1).cuda()

            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            outputs = net(inputs)
            # calculate the batch loss
            loss = criterion(outputs, targets)
            # backward pass
            loss.backward()
            # perform a single optimization step
            optimizer.step()
            # scheduler.step()
            # Update Train loss and accuracies
            trainLoss += loss.item()

        trainLoss = trainLoss / len(trainloader.sampler)
        training_losses.append(trainLoss)

        [t_kappa, t_loss, t_acc, t_conf] = testing(i, net, testloader, criterion)

        kappa_epoch.append(t_kappa)
        test_losses.append(t_loss)
        acc_epoch.append(t_acc)
        conf_matrix_all.append(t_conf)

    return (training_losses, kappa_epoch, test_losses, acc_epoch, conf_matrix_all)