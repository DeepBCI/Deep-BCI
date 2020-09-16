from data_loader import seedvig_loader
import torch
import torch.nn as nn
import pdb
import numpy as np
from model import dnn

from torch.autograd import Variable
import torch.optim

class Solver(object):
    def __init__(self, data_dir, batch_size, data_type):

        self.learning_rate = 0.0001
        self.beta1 = 0.5
        self.beta2 = 0.999
        self.max_epoch = 20
       

    def train(self):

        net = dnn().cuda()
        mse = nn.MSELoss()
        l1 = nn.L1Loss()
        opt = torch.optim.Adam(net.parameters(), self.learning_rate, [self.beta1, self.beta2])
        sigmoid = nn.Sigmoid().cuda()

        print('start train')

        train_err = []
        test_err_mse = []
        test_err_rmse = []

        prev_loss = 999.

        for sbj in range(5):
            traindata = seedvig_loader(data_dir, mode='train', batch_size=batch_size, test_no=sbj, data_type=data_type)
            testdata = seedvig_loader(data_dir, mode='val', batch_size=batch_size, test_no=sbj, data_type=data_type)
 
            for epoch in range(self.max_epoch):
    
                train_loss = 0.
                test_mse = 0.
                test_rmse = 0.
                n_train = 0.
                n_test = 0.
    
    
                net.train()  
                for i, data in enumerate(traindata):
                    eeg_x, eeg_y = Variable(data[0]).cuda(), Variable(data[1]).cuda()
                    pred = net(eeg_x)
                    err = torch.sqrt(mse(pred, eeg_y)) + l1(pred,eeg_y)
                    train_loss += err.item()
                    err.backward()
                    opt.step()
    
                if epoch==self.max_epoch-1:
                    train_err.append(train_loss/i)
                    n_train = i
                    prev_loss = train_loss
                net.eval()
        
                if epoch==self.max_epoch-1:
                    torch.save(net.state_dict(), '../model/dnn_3layers_{:03d}.ckpt'.format(sbj))
                    with torch.no_grad():
                        for i, data in enumerate(testdata):
                            eeg_x, eeg_y = Variable(data[0]).cuda(), Variable(data[1]).cuda()
                            pred = net(eeg_x)
                            err = mse(pred, eeg_y)
                            test_mse += err.item()
                            test_rmse += torch.sqrt(err).item()
                            
                    test_err_mse.append(test_mse/i)
                    test_err_rmse.append(test_rmse/i)
                    n_test = i
        
        print('mse: {:.4f} / mse(std): {:.4f}'.format(np.array(test_err_mse).mean(), np.array(test_err_mse).std()))
        print('rmse: {:.4f} / rmse(std): {:.4f}'.format(np.array(test_err_rmse).mean(), np.array(test_err_rmse).std()))


if __name__=='__main__':
    data_dir = '../data'
    batch_size = 256
    data_type = 'de_LDS'
    solver = Solver(data_dir, batch_size, data_type)
    solver.train()
