from vigloader_noon import seedvig_loader
import torch
import torch.nn as nn
import pdb
import numpy as np
from model import dnn_cls_da
import argparse

from torch.autograd import Variable
import torch.optim

torch.manual_seed(12)

class Solver(object):
    def __init__(self, data_dir, batch_size, data_type, lr, beta1, beta2, max_epoch, hdim, in_dim, dr, alpha, lambda_grl):

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.data_type = data_type
        self.learning_rate = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.max_epoch = max_epoch
        self.hdim = hdim
        self.in_dim = in_dim
        self.dr = dr
        self.alpha = alpha
        self.lambda_grl = lambda_grl   

    def train(self):

        ce = nn.CrossEntropyLoss()

        print('start train')
        test_acc = []
        prev_loss = 999.
        prev_acc = 0.
        b_acc = 0.
        b_out = 0.
        b_std = 0.
        b_epoch = 0.

        traindata = seedvig_loader(self.data_dir, mode='train', batch_size=self.batch_size, data_type=self.data_type, in_dim=self.in_dim)
        valdata = seedvig_loader(self.data_dir, mode='val', batch_size=self.batch_size, data_type=self.data_type, in_dim=self.in_dim)
        testdata = seedvig_loader(self.data_dir, mode='test', batch_size=self.batch_size, data_type=self.data_type, in_dim=self.in_dim)

        net = dnn_cls_da(hdim=self.hdim, in_dim=self.in_dim, dr=dr).cuda()
        opt = torch.optim.Adam(net.parameters(), self.learning_rate, [self.beta1, self.beta2], weight_decay=0.1)


        for epoch in range(self.max_epoch):

            train_loss = 0.
            n_train = 0.
            cor_tr = 0.
            n_tr = 0.
            
            net.train()  
            for i, data in enumerate(traindata):

                opt.zero_grad()

                eeg_x, eeg_y, eeg_s = Variable(data[0]).cuda(), Variable(data[1]).cuda(), Variable(data[2]).cuda()
                pred, pred_d = net(eeg_x, self.alpha)
                err_cls = ce(pred, eeg_y.view(eeg_y.shape[0]))
                
                err_sbj = ce(pred_d, eeg_s.view(eeg_s.shape[0]))
                err = err_cls + self.lambda_grl*err_sbj
                train_loss += (err_cls + self.lambda_grl*err_sbj)

                err.backward()
                opt.step()
            
                _, yhat = torch.max(pred, 1)
                cor_tr += (yhat==eeg_y.flatten()).sum().item()
                n_tr += eeg_x.shape[0]

            net.eval()
    
            with torch.no_grad():
                n_test = 0.
                test_err = 0.
                cor = 0.
                cor_all = np.array([])
                y_all = np.array([])
                sbj_all = np.array([])
 
                for i, data in enumerate(valdata):
                    eeg_x, eeg_y = Variable(data[0]).cuda(), Variable(data[1]).cuda()
                    pred, _ = net(eeg_x)
                    err = ce(pred, eeg_y.view(eeg_y.shape[0]))
                    test_err += err
                    n_test += eeg_x.shape[0]
                    
                    _, yhat = torch.max(pred, 1)
                    cor += (yhat == eeg_y.view(eeg_y.shape[0])).sum().item()
                    cor_all = np.concatenate((cor_all, yhat.cpu().numpy()))
                    y_all = np.concatenate((y_all, data[1].flatten().numpy()))
                    sbj_all = np.concatenate((sbj_all, data[2].flatten().numpy()))


                res = np.array([np.sum((sbj_all==elm)&(cor_all==y_all)) for elm in range(8)])
                nsbj = np.array([np.count_nonzero(sbj_all==elm) for elm in range(8)])
                out = res/nsbj
 
                print('Val {} err: {:.4f} / acc: {:.4f} / std(sbj): {:.4f} / #cor: {} / #val: {}'.format(epoch, test_err/i, cor/n_test, out.std(), cor, n_test))
                acc = cor/n_test




            if acc > prev_acc:
                prev_acc = acc
                n_test = 0.
                test_err = 0.
                cor = 0.
                cor_all = np.array([])
                y_all = np.array([])
                sbj_all = np.array([])
                
                torch.save(net.state_dict(), '../model.ckpt')

                with torch.no_grad():
                    for i, data in enumerate(testdata):
                        eeg_x, eeg_y = Variable(data[0]).cuda(), Variable(data[1]).cuda()
                        pred, _ = net(eeg_x)
                        err = ce(pred, eeg_y.view(eeg_y.shape[0]))
                        test_err += err
                        n_test += eeg_x.shape[0]
                        
                        _, yhat = torch.max(pred, 1)
                        cor += (yhat == eeg_y.view(eeg_y.shape[0])).sum().item()
                        cor_all = np.concatenate((cor_all, yhat.cpu().numpy()))
                        y_all = np.concatenate((y_all, data[1].flatten().numpy()))
                        sbj_all = np.concatenate((sbj_all, data[2].flatten().numpy()))

                    res = np.array([np.sum((sbj_all==elm)&(cor_all==y_all)) for elm in range(8)])
                    nsbj = np.array([np.count_nonzero(sbj_all==elm) for elm in range(8)])
                    out = res/nsbj

                    b_acc = cor/n_test
                    b_out = out
                    b_std = out.std()
                    b_epoch = epoch

        print('\nBest ACC: {:.4f} / STD: {:.4f}  at {} epochs'.format(b_acc, b_std, b_epoch))
        print(b_out)



if __name__=='__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='/seedvig_data_dir')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--data_type', type=str, default='de_LDS')
    parser.add_argument('--learning_rate', type=float, default=0.000001)
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--in_dim', type=int, default=425)
    parser.add_argument('--hdim', type=int, default=50)
    parser.add_argument('--dropout_rate', type=float, default=0.5)   
    parser.add_argument('--alpha', type=float, default=0.2)   
    parser.add_argument('--lambda_grl', type=float, default=0.5)   
 
    args = parser.parse_args()
    print(args)
     
    data_dir = args.data_dir
    batch_size = args.batch_size
    data_type = args.data_type
    lr = args.learning_rate
    max_epoch = args.max_epoch
    beta1 = args.beta1
    beta2 = args.beta2
    hdim = args.hdim
    in_dim = args.in_dim
    dr = args.dropout_rate
    alpha = args.alpha
    lambda_grl = args.lambda_grl

    solver = Solver(data_dir, batch_size, data_type, lr, beta1, beta2, max_epoch, hdim, in_dim, dr, alpha, lambda_grl)
    solver.train()
