
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from Network import Unet1D
from Network.Unet2D import UNet
from tqdm import tqdm
import Training as Training
import DataPreprocessing as DP
import torch.optim as optim

## gpu setting ##
DP.gpu_setting(gpu_num=0)

## data load ##
[data, label] = DP.load_data('Validation set')

## load pretrained model ##
pre_mdl = UNet(1, 5)
chk = torch.load('Unet2D\\epoch50_all')
pre_mdl.load_state_dict(chk['model_state_dict'])
pre_mdl.eval()

results_dir = './spectrogram/Validation set/'
[spectrogram, labels] = DP.load_data(results_dir)

activation = {'len': spectrogram['len']}

for i in tqdm(range(spectrogram['len']), desc='Activation '):
    input = np.transpose(spectrogram[i], [2, 0, 1])
    input = torch.FloatTensor(input)
    input = input.unsqueeze(1)
    output = pre_mdl.extract_feature(input)
    output = output.view(output.shape[0], -1)
    activation[i] = output.detach().numpy()


acc_all = np.zeros(10)
kappa_all = np.zeros(10)
for test_sub in range(10):

    ## initialize variable ##
    start_epoch = 0
    max_epoch = 100
    mdl = Unet1D.UNet_spectrogram(1, 2)
    mdl.cuda()
    loss = nn.CrossEntropyLoss().cuda()
    batch_size = 8
    optimizer = optim.Adam(mdl.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-7)

    [trainx, trainy, valx, valy] = DP.train_val_split(data, label, test_sub)
    [train_act, trainy, val_act, valy] = DP.train_val_split(activation, labels, test_sub)

    valx = torch.FloatTensor(valx).cuda()
    val_act = torch.FloatTensor(val_act).cuda()
    valy = torch.LongTensor(valy.reshape(-1)).cuda()

    trainx = torch.FloatTensor(trainx).cuda()
    train_act = torch.FloatTensor(train_act).cuda()
    trainy = torch.LongTensor(trainy.reshape(-1))

    setting = {'data': [trainx, trainy, valx, valy],
               'spectro': [train_act, trainy, val_act, valy],
               'mdl': mdl,
               'optim': optimizer,
               'criterion': loss,
               'max_epochs': max_epoch,
               'fold_idx': test_sub,
               'batch_size': batch_size,
               'output_dir': 'Ours\\Utime\\'}

    [loss_list, t_acc_list, v_acc_list, t_kappa_list, v_kappa_list] = Training.training_spectro(setting)

    x_scale = np.arange(max_epoch)
    plt.figure()
    plt.xlim([0, max_epoch])
    plt.plot(x_scale, loss_list, c='red')
    plt.plot(x_scale, t_acc_list, c='blue')
    plt.plot(x_scale, v_acc_list, c='green')
    plt.plot(x_scale, t_kappa_list)
    plt.plot(x_scale, v_kappa_list)
    plt.legend(['Train loss', 'Train acc', 'Valid acc', 'Train kappa', 'Valid kappa'])
    plt.savefig('Ours\\Utime\\training' + str(test_sub) + '.png')

    print('[MAX Scores of Fold{}] [TRAIN] acc:{:.3f} kappa:{:.3f} // [VALID] acc:{:.3f}, kappa:{:.3f}'
          .format(test_sub + 1, np.max(t_acc_list), np.max(t_kappa_list), np.max(v_acc_list), np.max(v_kappa_list)))

    acc_all[test_sub] = np.max(v_acc_list)
    kappa_all[test_sub] = np.max(v_kappa_list)

print('[Mean ACC] %.3f (%.3f)/ [Mean Kappa] %.3f (%.3f)'
      %(np.mean(acc_all), np.std(acc_all), np.mean(kappa_all), np.std(kappa_all)))
