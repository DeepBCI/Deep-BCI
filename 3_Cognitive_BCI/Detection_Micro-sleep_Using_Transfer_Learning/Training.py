import torch
import math
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from tqdm import tqdm

def training(setting):
    loss_list = []
    t_kappa_list = []
    t_acc_list = []
    v_acc_list = []
    v_kappa_list = []

    [train_x, train_y, valid_x, valid_y] = setting['data']
    mdl = setting['mdl']
    optim = setting['optim']
    criterion = setting['criterion']
    max_epochs = setting['max_epochs']
    fold_idx = setting['fold_idx']
    batch_size = setting['batch_size']
    output_dir = setting['output_dir']

    valid_data = TensorDataset(valid_x, valid_y)
    max_kappa = -999
    for epoch in tqdm(range(max_epochs), desc='Training  '+str(fold_idx +1)):
        loss_sum = 0.0
        train_y = train_y.cuda()
        train_data = TensorDataset(train_x, train_y)
        train_data_ = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False)

        all_pred_list = []
        all_v_pred_list = []
        for j, train in enumerate(train_data_):
            input, label = train
            input = input.view(-1, 1, 3000)

            mdl.zero_grad()

            pred_y = mdl(input)

            loss = criterion(pred_y, label)

            loss.backward()
            optim.step()
            loss_sum += loss

            pred_list = torch.max(pred_y, dim=1)
            pred_list = pred_list[1].cpu()
            pred_list = pred_list.detach().numpy()

            all_pred_list.append(pred_list)

        all_pred = np.concatenate(all_pred_list)
        all_pred = torch.tensor(all_pred)

        train_y = train_y.cpu()
        acc = int(sum(all_pred == train_y))
        acc = acc / train_x.shape[0]

        t_acc_list.append(acc)
        kappa = cohen_kappa_score(all_pred, train_y)
        t_kappa_list.append(kappa)
        if math.isnan(kappa):
            kappa = 0.0
        loss_sum = loss_sum / train_x.shape[0]
        loss_list.append(loss_sum)

        with torch.no_grad():
            mdl.eval()
            for k, valid_ in enumerate(valid_data):
                v_input, v_label = valid_
                v_input = v_input.view(-1, 1, 3000)

                v_pred_y = mdl(v_input)

                v_pred_list = torch.max(v_pred_y, dim=1)
                v_pred_list = v_pred_list[1].cpu()
                v_pred_list = v_pred_list.detach().numpy()

                all_v_pred_list.append(v_pred_list)

            all_v_pred = np.concatenate(all_v_pred_list)
            all_v_pred = torch.tensor(all_v_pred)
            valid_y = valid_y.cpu()
            v_kappa = cohen_kappa_score(all_v_pred, valid_y)

            if math.isnan(v_kappa):
                v_kappa = 0.0

            v_acc = int(sum(all_v_pred == valid_y))
            v_acc = v_acc / valid_x.shape[0]
            """
            print('[Fold{}, Epoch{}] [TRAIN] loss:{:.3f} // [Valid] acc:{:.3f}, kappa:{:.3f}'
                .format(fold_idx + 1, epoch + 1, loss_sum, v_acc, v_kappa))
            """
            if v_kappa > max_kappa:
                max_kappa = v_kappa
                torch.save({
                    'epoch': epoch,
                    'v_kappa': v_kappa,
                    'valid_x': valid_x,
                    'valid_y': valid_y,
                    'model_state_dict': mdl.state_dict(),
                    'optimizer_state_dict': optim.state_dict()
                }, output_dir + 'Fold{}.pth'.format(fold_idx + 1))

            v_acc_list.append(v_acc)
            v_kappa_list.append(v_kappa)

            mdl.train()

    return (loss_list, t_acc_list, v_acc_list, t_kappa_list, v_kappa_list)

def training_spectro(setting):
    loss_list = []
    t_kappa_list = []
    t_acc_list = []
    v_acc_list = []
    v_kappa_list = []

    [train_x, train_y, valid_x, valid_y] = setting['data']
    [train_act, train_y, valid_act, valid_y] = setting['spectro']
    mdl = setting['mdl']
    optim = setting['optim']
    criterion = setting['criterion']
    max_epochs =  setting['max_epochs']
    fold_idx = setting['fold_idx']
    batch_size = setting['batch_size']
    output_dir = setting['output_dir']

    valid_data = TensorDataset(valid_x, valid_act, valid_y)
    valid_data_ = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=False)

    max_kappa = -999
    for epoch in tqdm(range(max_epochs), desc='Training  '+str(fold_idx +1)):
        loss_sum = 0.0
        train_y = train_y.cuda()
        train_data = TensorDataset(train_x, train_act, train_y)
        train_data_ = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False)

        all_pred_list = []
        all_v_pred_list = []
        for j, train in enumerate(train_data_):
            input, spectro, label = train
            input = input.view(-1, 1, 3000)

            mdl.zero_grad()

            pred_y = mdl(input, spectro)

            loss = criterion(pred_y, label)

            loss.backward()
            optim.step()
            loss_sum += loss

            pred_list = torch.max(pred_y, dim=1)
            pred_list = pred_list[1].cpu()
            pred_list = pred_list.detach().numpy()

            all_pred_list.append(pred_list)

        all_pred = np.concatenate(all_pred_list)
        all_pred = torch.tensor(all_pred)

        train_y = train_y.cpu()
        acc = int(sum(all_pred == train_y))
        acc = acc / train_x.shape[0]

        t_acc_list.append(acc)
        kappa = cohen_kappa_score(all_pred, train_y)
        t_kappa_list.append(kappa)
        if math.isnan(kappa):
            kappa = 0.0
        loss_sum = loss_sum / train_x.shape[0]
        loss_list.append(loss_sum)

        with torch.no_grad():
            mdl.eval()
            for k, valid_ in enumerate(valid_data_):
                v_input, v_spectro, v_label = valid_
                v_input = v_input.view(-1, 1, 3000)

                v_pred_y = mdl(v_input, v_spectro)

                v_pred_list = torch.max(v_pred_y, dim=1)
                v_pred_list = v_pred_list[1].cpu()
                v_pred_list = v_pred_list.detach().numpy()

                all_v_pred_list.append(v_pred_list)

            all_v_pred = np.concatenate(all_v_pred_list)
            all_v_pred = torch.tensor(all_v_pred)
            valid_y = valid_y.cpu()
            v_kappa = cohen_kappa_score(all_v_pred, valid_y)

            if math.isnan(v_kappa):
                v_kappa = 0.0

            v_acc = int(sum(all_v_pred == valid_y))
            v_acc = v_acc / valid_x.shape[0]
            """
            print('[Fold{}, Epoch{}] [TRAIN] loss:{:.3f} // [Valid] acc:{:.3f}, kappa:{:.3f}'
                .format(fold_idx + 1, epoch + 1, loss_sum, v_acc, v_kappa))
            """
            if v_kappa > max_kappa:
                max_kappa = v_kappa
                torch.save({
                    'epoch': epoch,
                    'v_kappa': v_kappa,
                    'model_state_dict': mdl.state_dict(),
                    'optimizer_state_dict': optim.state_dict()
                }, output_dir + 'Fold{}.pth'.format(fold_idx + 1))

            v_acc_list.append(v_acc)
            v_kappa_list.append(v_kappa)
            mdl.train()

    return (loss_list, t_acc_list, v_acc_list, t_kappa_list, v_kappa_list)

def training_dfn(setting):
    loss_list = []
    t_kappa_list = []
    t_acc_list = []
    v_acc_list = []
    v_kappa_list = []

    [train_x, train_y, valid_x, valid_y] = setting['data']
    mdl = setting['mdl']
    optim = setting['optim']
    criterion = setting['criterion']
    max_epochs =  setting['max_epochs']
    fold_idx = setting['fold_idx']
    batch_size = setting['batch_size']
    output_dir = setting['output_dir']

    valid_data = TensorDataset(valid_x, valid_y)
    valid_data_ = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=False)
    max_kappa = -9999

    for epoch in tqdm(range(max_epochs), desc='Training DFN '+str(fold_idx +1)):
        loss_sum = 0.0
        train_y = train_y.cuda()
        train_data = TensorDataset(train_x, train_y)
        train_data_ = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False)

        all_pred_list = []
        all_v_pred_list = []

        t_res_list = []
        v_res_list = []

        t_label_list = []
        v_label_list = []

        for j, train in enumerate(train_data_):
            input, label = train
            input = input.view(-1, 1, 3000)

            mdl.zero_grad()

            pred_y, res = mdl(input)
            t_res_list += [res]
            t_label_list += [label]

            loss = criterion(pred_y, label)

            loss.backward()
            optim.step()
            loss_sum += loss

            pred_list = torch.max(pred_y, dim=1)
            pred_list = pred_list[1].cpu()
            pred_list = pred_list.detach().numpy()

            all_pred_list.append(pred_list)

        all_pred = np.concatenate(all_pred_list)
        all_pred = torch.tensor(all_pred)

        train_y = train_y.cpu()
        acc = int(sum(all_pred == train_y))
        acc = acc / train_x.shape[0]

        t_acc_list.append(acc)
        kappa = cohen_kappa_score(all_pred, train_y)
        t_kappa_list.append(kappa)
        if math.isnan(kappa):
            kappa = 0.0
        loss_sum = loss_sum / train_x.shape[0]
        loss_list.append(loss_sum)

        with torch.no_grad():
            mdl.eval()
            for k, valid in enumerate(valid_data_):
                v_input, v_label = valid
                v_input = v_input.view(-1, 1, 3000)

                v_pred_y, v_res = mdl(v_input)
                v_res_list += [v_res]
                v_label_list += [v_label]

                v_pred_list = torch.max(v_pred_y, dim=1)
                v_pred_list = v_pred_list[1].cpu()
                v_pred_list = v_pred_list.detach().numpy()

                all_v_pred_list.append(v_pred_list)

            all_v_pred = np.concatenate(all_v_pred_list)
            all_v_pred = torch.tensor(all_v_pred)
            valid_y = valid_y.cpu()
            v_kappa = cohen_kappa_score(all_v_pred, valid_y)

            if math.isnan(v_kappa):
                v_kappa = 0.0

            v_acc = int(sum(all_v_pred == valid_y))
            v_acc = v_acc / valid_x.shape[0]
            """
            print('[Fold{}, Epoch{}] [TRAIN] loss:{:.3f} // [Valid] acc:{:.3f}, kappa:{:.3f}'
                .format(fold_idx + 1, epoch + 1, loss_sum, v_acc, v_kappa))
            """

            if v_kappa > max_kappa:
                max_kappa = v_kappa
                torch.save({
                    'epoch': epoch,
                    'v_kappa': v_kappa,
                    'model_state_dict': mdl.state_dict(),
                    'optimizer_state_dict': optim.state_dict(),
                    't_res': t_res_list,
                    't_label': t_label_list,
                    'v_res': v_res_list,
                    'v_label': v_label_list
                }, output_dir + 'Fold{}.pth'.format(fold_idx + 1))

            v_acc_list.append(v_acc)
            v_kappa_list.append(v_kappa)
            mdl.train()

    return (loss_list, t_acc_list, v_acc_list,
            t_kappa_list, v_kappa_list)

def training_dfn_spectro(setting):
    loss_list = []
    t_kappa_list = []
    t_acc_list = []
    v_acc_list = []
    v_kappa_list = []

    [train_x, train_y, valid_x, valid_y] = setting['data']
    [train_acti, train_y, valid_acti, valid_y] = setting['acti']
    mdl = setting['mdl']
    optim = setting['optim']
    criterion = setting['criterion']
    max_epochs =  setting['max_epochs']
    fold_idx = setting['fold_idx']
    batch_size = setting['batch_size']
    output_dir = setting['output_dir']

    valid_data = TensorDataset(valid_x, valid_acti, valid_y)
    valid_data_ = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=False)
    max_kappa = -9999

    for epoch in tqdm(range(max_epochs), desc='Training DFN '+str(fold_idx +1)):
        loss_sum = 0.0
        train_y = train_y.cuda()
        train_data = TensorDataset(train_x, train_acti, train_y)
        train_data_ = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False)

        all_pred_list = []
        all_v_pred_list = []

        t_res_list = []
        v_res_list = []

        t_label_list = []
        v_label_list = []

        for j, train in enumerate(train_data_):
            input, spectro, label = train
            input = input.view(-1, 1, 3000)

            mdl.zero_grad()

            pred_y, res = mdl(input, spectro)
            t_res_list += [res]
            t_label_list += [label]

            loss = criterion(pred_y, label)

            loss.backward()
            optim.step()
            loss_sum += loss

            pred_list = torch.max(pred_y, dim=1)
            pred_list = pred_list[1].cpu()
            pred_list = pred_list.detach().numpy()

            all_pred_list.append(pred_list)

        all_pred = np.concatenate(all_pred_list)
        all_pred = torch.tensor(all_pred)

        train_y = train_y.cpu()
        acc = int(sum(all_pred == train_y))
        acc = acc / train_x.shape[0]

        t_acc_list.append(acc)
        kappa = cohen_kappa_score(all_pred, train_y)
        t_kappa_list.append(kappa)
        if math.isnan(kappa):
            kappa = 0.0
        loss_sum = loss_sum / train_x.shape[0]
        loss_list.append(loss_sum)

        with torch.no_grad():
            mdl.eval()
            for k, valid in enumerate(valid_data_):
                v_input, v_spectro, v_label = valid
                v_input = v_input.view(-1, 1, 3000)

                v_pred_y, v_res = mdl(v_input, v_spectro)
                v_res_list += [v_res]
                v_label_list += [v_label]

                v_pred_list = torch.max(v_pred_y, dim=1)
                v_pred_list = v_pred_list[1].cpu()
                v_pred_list = v_pred_list.detach().numpy()

                all_v_pred_list.append(v_pred_list)

            all_v_pred = np.concatenate(all_v_pred_list)
            all_v_pred = torch.tensor(all_v_pred)
            valid_y = valid_y.cpu()
            v_kappa = cohen_kappa_score(all_v_pred, valid_y)

            if math.isnan(v_kappa):
                v_kappa = 0.0

            v_acc = int(sum(all_v_pred == valid_y))
            v_acc = v_acc / valid_x.shape[0]
            """
            print('[Fold{}, Epoch{}] [TRAIN] loss:{:.3f} // [Valid] acc:{:.3f}, kappa:{:.3f}'
                .format(fold_idx + 1, epoch + 1, loss_sum, v_acc, v_kappa))
            """

            if v_kappa > max_kappa:
                max_kappa = v_kappa
                torch.save({
                    'epoch': epoch,
                    'v_kappa': v_kappa,
                    'model_state_dict': mdl.state_dict(),
                    'optimizer_state_dict': optim.state_dict(),
                    't_res': t_res_list,
                    't_label': t_label_list,
                    'v_res': v_res_list,
                    'v_label': v_label_list
                }, output_dir + 'Fold{}.pth'.format(fold_idx + 1))

            v_acc_list.append(v_acc)
            v_kappa_list.append(v_kappa)
            mdl.train()

    return (loss_list, t_acc_list, v_acc_list,
            t_kappa_list, v_kappa_list)

def training_seq(setting):
    loss_list = []
    t_kappa_list = []
    t_acc_list = []
    v_acc_list = []
    v_kappa_list = []

    [train_x, train_y, valid_x, valid_y] = setting['data']
    mdl = setting['mdl']
    optim = setting['optim']
    criterion = setting['criterion']
    max_epochs = setting['max_epochs']
    fold_idx = setting['fold_idx']
    batch_size = setting['batch_size']
    output_dir = setting['output_dir']

    valid_data = TensorDataset(valid_x, valid_y)
    valid_data_ = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=False)
    max_kappa = -9999

    for epoch in tqdm(range(max_epochs), desc='Training SEQ '+str(fold_idx +1)):
        loss_sum = 0.0
        train_y = train_y.cuda()
        train_data = TensorDataset(train_x, train_y)
        train_data_ = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False)

        all_pred_list = []
        all_v_pred_list = []
        for j, train in enumerate(train_data_):
            input, label = train

            mdl.zero_grad()
            input = input.unsqueeze(1)

            pred_y = mdl(input)
            pred_y = pred_y.view(pred_y.shape[0], -1)
            loss = criterion(pred_y, label)

            loss.backward()
            optim.step()
            loss_sum += loss

            pred_list = torch.max(pred_y, dim=1)
            pred_list = pred_list[1].cpu()
            pred_list = pred_list.detach().numpy()

            all_pred_list.append(pred_list)

        all_pred = np.concatenate(all_pred_list)
        all_pred = torch.tensor(all_pred)

        train_y = train_y.cpu()
        acc = int(sum(all_pred == train_y))
        acc = acc / train_x.shape[0]

        t_acc_list.append(acc)
        kappa = cohen_kappa_score(all_pred, train_y)
        t_kappa_list.append(kappa)
        if math.isnan(kappa):
            kappa = 0.0
        loss_sum = loss_sum / train_x.shape[0]
        loss_list.append(loss_sum)

        with torch.no_grad():
            mdl.eval()
            for k, valid_ in enumerate(valid_data_):
                v_input, v_label = valid_
                v_input = v_input.unsqueeze(1)

                v_pred_y = mdl(v_input)

                v_pred_y = v_pred_y.view(v_pred_y.shape[0], -1)
                v_pred_list = torch.max(v_pred_y, dim=1)
                v_pred_list = v_pred_list[1].cpu()
                v_pred_list = v_pred_list.detach().numpy()

                all_v_pred_list.append(v_pred_list)

            all_v_pred = np.concatenate(all_v_pred_list)
            all_v_pred = torch.tensor(all_v_pred)
            valid_y = valid_y.cpu()
            v_kappa = cohen_kappa_score(all_v_pred, valid_y)

            if math.isnan(v_kappa):
                v_kappa = 0.0

            v_acc = int(sum(all_v_pred == valid_y))
            v_acc = v_acc / valid_x.shape[0]
            """
            print('[Fold{}, Epoch{}] [TRAIN] loss:{:.3f} // [Valid] acc:{:.3f}, kappa:{:.3f}'
                .format(fold_idx + 1, epoch + 1, loss_sum, v_acc, v_kappa))
            """

            if v_kappa > max_kappa:
                max_kappa = v_kappa
                torch.save({
                    'epoch': epoch,
                    'v_kappa': v_kappa,
                    'model_state_dict': mdl.state_dict(),
                    'optimizer_state_dict': optim.state_dict()
                }, output_dir + 'Fold{}_SEQ.pth'.format(fold_idx + 1))

            v_acc_list.append(v_acc)
            v_kappa_list.append(v_kappa)
            mdl.train()

    return (loss_list, t_acc_list, v_acc_list,
            t_kappa_list, v_kappa_list)

def training_seq_spectro(setting):
    loss_list = []
    t_kappa_list = []
    t_acc_list = []
    v_acc_list = []
    v_kappa_list = []

    [train_x, train_y, valid_x, valid_y] = setting['data']
    [train_acti, train_y, valid_acti, valid_y] = setting['acti']
    mdl = setting['mdl']
    optim = setting['optim']
    criterion = setting['criterion']
    max_epochs = setting['max_epochs']
    fold_idx = setting['fold_idx']
    batch_size = setting['batch_size']
    output_dir = setting['output_dir']

    valid_data = TensorDataset(valid_x, valid_acti, valid_y)
    valid_data_ = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=False)

    max_kappa = -9999

    for epoch in tqdm(range(max_epochs), desc='Training SEQ '+str(fold_idx +1)):
        loss_sum = 0.0
        train_y = train_y.cuda()
        train_data = TensorDataset(train_x, train_acti, train_y)
        train_data_ = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False)

        all_pred_list = []
        all_v_pred_list = []
        for j, train in enumerate(train_data_):
            input, spectro, label = train

            mdl.zero_grad()
            input = input.unsqueeze(1)

            pred_y = mdl(input, spectro)
            pred_y = pred_y.view(pred_y.shape[0], -1)
            loss = criterion(pred_y, label)

            loss.backward()
            optim.step()
            loss_sum += loss

            pred_list = torch.max(pred_y, dim=1)
            pred_list = pred_list[1].cpu()
            pred_list = pred_list.detach().numpy()

            all_pred_list.append(pred_list)

        all_pred = np.concatenate(all_pred_list)
        all_pred = torch.tensor(all_pred)

        train_y = train_y.cpu()
        acc = int(sum(all_pred == train_y))
        acc = acc / train_x.shape[0]

        t_acc_list.append(acc)
        kappa = cohen_kappa_score(all_pred, train_y)
        t_kappa_list.append(kappa)
        if math.isnan(kappa):
            kappa = 0.0
        loss_sum = loss_sum / train_x.shape[0]
        loss_list.append(loss_sum)

        with torch.no_grad():
            mdl.eval()
            for k, valid_ in enumerate(valid_data_):
                v_input, v_spectro, v_label = valid_
                v_input = v_input.unsqueeze(1)

                v_pred_y = mdl(v_input, v_spectro)

                v_pred_y = v_pred_y.view(v_pred_y.shape[0], -1)
                v_pred_list = torch.max(v_pred_y, dim=1)
                v_pred_list = v_pred_list[1].cpu()
                v_pred_list = v_pred_list.detach().numpy()

                all_v_pred_list.append(v_pred_list)

            all_v_pred = np.concatenate(all_v_pred_list)
            all_v_pred = torch.tensor(all_v_pred)
            valid_y = valid_y.cpu()
            v_kappa = cohen_kappa_score(all_v_pred, valid_y)

            if math.isnan(v_kappa):
                v_kappa = 0.0

            v_acc = int(sum(all_v_pred == valid_y))
            v_acc = v_acc / valid_x.shape[0]

            if v_kappa > max_kappa:
                max_kappa = v_kappa
                torch.save({
                    'epoch': epoch,
                    'v_kappa': v_kappa,
                    'model_state_dict': mdl.state_dict(),
                    'optimizer_state_dict': optim.state_dict()
                }, output_dir + 'Fold{}_SEQ.pth'.format(fold_idx + 1))

            v_acc_list.append(v_acc)
            v_kappa_list.append(v_kappa)
            mdl.train()

    return (loss_list, t_acc_list, v_acc_list,
            t_kappa_list, v_kappa_list)