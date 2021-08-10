import numpy as np
import torch
import torch.nn as nn
from Network import Unet2D
import DataPreprocessing as DP
import math
import torch.optim as optim
from tqdm import tqdm

[data, label] = DP.load_data('spectrogram\\Training set')

## gpu setting ##
DP.gpu_setting(gpu_num=0)


for test_sub in range(0, 50):
    ## initialize variable ##
    start_epoch = 0
    max_epoch = 50
    mdl = []
    mdl = Unet2D.UNet(1, 5)
    mdl.cuda()
    loss = nn.CrossEntropyLoss()
    batch_size = 128
    optimizer = optim.Adam(mdl.parameters(), lr=1e-5, betas=(0.9, 0.999), weight_decay=0.01)

    ## training model ##
    for epoch in range(start_epoch, max_epoch):

        if not (start_epoch == 0):
            checkpoint = torch.load('Unet2D\\epoch'+str(start_epoch)+'_test_sub'+str(test_sub+1))
            mdl.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            loss = checkpoint['loss']
            mdl.train()
        loss_epoch = 0.0

        for sub in tqdm(range(data['len'])):

            if sub == test_sub:
                continue

            n = data[sub].shape[2]
            batch_max = math.ceil(n/batch_size)
            temp = data[sub]
            tempy = label[sub]
            temp = np.transpose(temp, [2, 0, 1])
            idx = np.arange(n)
            np.random.shuffle(idx)
            temp = temp[idx, :, :]
            tempy = tempy[:, idx]
            loss_tot = 0.0

            for batch_num in range(batch_max):

                xtrain = torch.FloatTensor(temp[batch_size*batch_num:batch_size*(batch_num+1), :, :]).cuda()
                ytrain = torch.LongTensor(np.squeeze(tempy[:, batch_size*batch_num:batch_size*(batch_num+1)])).cuda()

                xtrain = xtrain.unsqueeze(1)

                out = mdl(xtrain)

                loss_sub = loss(out, ytrain)
                mdl.zero_grad()
                loss_sub.backward()
                optimizer.step()

                loss_tot += loss_sub.data.item()

            #print('[Epoch %d/Sub %d] loss: %.3f'%(epoch+1, sub+1, loss_tot))
            loss_epoch += loss_tot

        print('[Epoch %d] total loss: %.3f'%(epoch+1, loss_epoch))

        torch.save({
            'model_state_dict': mdl.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }, 'Unet2D\\epoch'+str(epoch+1)+'_test_sub'+str(test_sub+1))

