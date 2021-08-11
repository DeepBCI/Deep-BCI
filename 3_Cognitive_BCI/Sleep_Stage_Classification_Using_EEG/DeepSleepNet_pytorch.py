import os
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, confusion_matrix
import torch.nn.functional as F
from tqdm import tqdm
from scipy import io
from dataset_split import train_data_split
import argparse

class DeepFeatureNet(nn.Module):
    def __init__(self, sampling_rate=125):
        super(DeepFeatureNet, self).__init__()
        self.conv1d_s = nn.Sequential(nn.Conv1d(1, 64, kernel_size=sampling_rate//2, stride=sampling_rate//16, padding=sampling_rate//4, bias=False),
                                    nn.BatchNorm1d(64, momentum=0.001, eps=1e-5), nn.ReLU(), nn.MaxPool1d(kernel_size=8, stride=8, padding=4), nn.Dropout(),
                                    nn.Conv1d(in_channels=64, out_channels=128, kernel_size=8, stride=1, padding=4, bias=False),  nn.BatchNorm1d(128, momentum=0.001, eps=1e-5), nn.ReLU(),
                                    nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8, stride=1, padding=4, bias=False), nn.BatchNorm1d(128, momentum=0.001, eps=1e-5), nn.ReLU(),
                                    nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8, stride=1, padding=4, bias=False), nn.BatchNorm1d(128, momentum=0.001, eps=1e-5), nn.ReLU(),
                                    nn.MaxPool1d(kernel_size=4, stride=4, padding=2))
        self.conv1d_l = nn.Sequential(nn.Conv1d(1, 64, kernel_size=sampling_rate*4, stride=sampling_rate//2, padding=sampling_rate*2, bias=False),
                                    nn.BatchNorm1d(64, momentum=0.001, eps=1e-5), nn.ReLU(), nn.MaxPool1d(kernel_size=4, stride=4, padding=2), nn.Dropout(),
                                    nn.Conv1d(in_channels=64, out_channels=128, kernel_size=6, stride=1, padding=3, bias=False),  nn.BatchNorm1d(128, momentum=0.001, eps=1e-5), nn.ReLU(),
                                    nn.Conv1d(in_channels=128, out_channels=128, kernel_size=6, stride=1, padding=3, bias=False), nn.BatchNorm1d(128, momentum=0.001, eps=1e-5), nn.ReLU(),
                                    nn.Conv1d(in_channels=128, out_channels=128, kernel_size=6, stride=1, padding=3, bias=False), nn.BatchNorm1d(128, momentum=0.001, eps=1e-5), nn.ReLU(),
                                    nn.MaxPool1d(kernel_size=2, stride=2, padding=1))
        self.fc = nn.Sequential(nn.Linear(3584, n_classes, bias=False))
        self.droppout = nn.Dropout()
    def forward(self, data):
        input1 = data
        input2 = data
        x1 = self.conv1d_s(input1)
        x2 = self.conv1d_l(input2)
        res = torch.cat((x1.flatten(1), x2.flatten(1)), dim=1)
        pred = self.fc(res)
        res = self.droppout(res)
        return pred, res

class DeepSleepNet(nn.Module):
    def __init__(self, input_size=3584, hidden_size = 512, n_rnn_layers=2, n_classes=5, bidirectional=True):
        super(DeepSleepNet, self).__init__()
        self.n_rnn_layers = n_rnn_layers
        if bidirectional:
            self.n_directions = 2
        else:
            self.n_directions = 1
        self.hidden_size = hidden_size
        self.res = nn.Sequential(nn.Linear(input_size, 1024, bias=False), nn.BatchNorm1d(1024, momentum=0.001, eps=1e-5), nn.ReLU())
        self.dropout = nn.Dropout()
        self.lstm = nn.LSTM(input_size, hidden_size, n_rnn_layers, dropout=0.5,
                            bidirectional=bidirectional, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size*self.n_directions, hidden_size, n_rnn_layers, dropout=0.5,
                             bidirectional=bidirectional, batch_first=True)
        self.output = nn.Linear(hidden_size*self.n_directions, n_classes, bias=False)

    def forward(self, input, hidden, cell):
        b, s, l = input.shape
        x = input
        res = input
        res = self.res(res.view(-1, l))
        x, state = self.lstm(x, (hidden, cell))
        x, state = self.lstm2(x, state)
        result = x + res.view(b,s,-1)
        result = self.dropout(result)
        result = self.output(result)
        return result

    def init_state(self, batch_size):
        hidden = torch.zeros(self.n_rnn_layers * self.n_directions, batch_size, self.hidden_size)
        cell = torch.zeros(self.n_rnn_layers * self.n_directions, batch_size, self.hidden_size)
        return hidden, cell

def preprocess_data(path, filename):
    f = io.loadmat(path + filename)
    out = f.get('psg')
    return out

def load_header(path, filename):
    f = io.loadmat(path + filename)
    out = f.get('hyp')[0]
    return out

class CustomDataset(Dataset):
    def __init__(self, psg, hyp, istrain, is1, seq_len):
        self.x_data = []
        self.y_data = []
        for i in range(len(psg)):
            if is1:
                if istrain:
                    self.x_data.extend(psg[i])
                    self.y_data.extend(hyp[i])
                else:
                    self.x_data.extend(psg[i])
                    self.y_data.extend(hyp[i])
            else:
                psg_tmp = []
                hyp_tmp = []
                if istrain:
                    for j in range(len(psg[i]) - seq_len + 1):
                        psg_tmp.append(psg[i][j:j + seq_len])
                        hyp_tmp.append(hyp[i][j:j + seq_len])
                    self.x_data.extend(psg_tmp)
                    self.y_data.extend(hyp_tmp)
                else:
                    for j in range(len(psg[i]) - seq_len + 1):
                        psg_tmp.append(psg[i][j:j + seq_len])
                        hyp_tmp.append(hyp[i][j:j + seq_len])
                    self.x_data.extend(psg_tmp)
                    self.y_data.extend(hyp_tmp)
        self.x = torch.FloatTensor(self.x_data)
        self.y = torch.LongTensor(self.y_data)
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        return x, y

class dataset(Dataset):
    def __init__(self, x_list, y_list):
        super(dataset, self).__init__()
        x = torch.empty(0)
        y = torch.empty(0)
        print('Loading dataset...')
        for i in tqdm(range(0, len(x_list))):
            x = torch.cat((x, torch.tensor(x_list[i], dtype=torch.float)))
            y = torch.cat((y, torch.tensor(y_list[i], dtype=torch.long)))
        b, c, i = x.shape
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)

def get_balance_class_downsample(x, y):
    x = x.numpy()
    y = y.numpy()
    class_labels = np.unique(y)
    n_min_classes = -1
    for c in class_labels:
        n_samples = len(np.where(y == c)[0])
        if n_min_classes == -1:
            n_min_classes = n_samples
        elif n_min_classes > n_samples:
            n_min_classes = n_samples
    balance_x = []
    balance_y = []
    for c in class_labels:
        idx = np.where(y == c)[0]
        idx = np.random.permutation(idx)[:n_min_classes]
        balance_x.append(x[idx])
        balance_y.append(y[idx])
    balance_x = np.vstack(balance_x)
    balance_y = np.hstack(balance_y)
    return torch.tensor(balance_x), torch.tensor(balance_y)

def get_balance_class_oversample(x, y):
    x = x.numpy()
    y = y.numpy()
    class_labels = np.unique(y)
    n_max_classes = -1
    for c in class_labels:
        n_samples = len(np.where(y == c)[0])
        if n_max_classes < n_samples:
            n_max_classes = n_samples
    balance_x = []
    balance_y = []
    for c in class_labels:
        idx = np.where(y == c)[0]
        n_samples = len(idx)
        n_repeats = int(n_max_classes / n_samples)
        tmp_x = np.repeat(x[idx], n_repeats, axis=0)
        tmp_y = np.repeat(y[idx], n_repeats, axis=0)
        n_remains = n_max_classes - len(tmp_x)
        if n_remains > 0:
            sub_idx = np.random.permutation(idx)[:n_remains]
            tmp_x = np.vstack([tmp_x, x[sub_idx]])
            tmp_y = np.hstack([tmp_y, y[sub_idx]])
        balance_x.append(tmp_x)
        balance_y.append(tmp_y)
    balance_x = np.vstack(balance_x)
    balance_y = np.hstack(balance_y)
    return torch.tensor(balance_x), torch.tensor(balance_y)

def weighted_loss(model_output, true_label, cf):
    out = 0
    for i, item in enumerate(model_output):
        item2 = torch.unsqueeze(item, 0)
        t = torch.unsqueeze(true_label[i], 0)
        if model_output[i].argmax() == true_label[i]:
            w = 1
        else:
            if cf[true_label[i]][model_output[i].argmax()] < 0.01:
                w = 1
            else:
                w = 100 * cf[true_label[i]][model_output[i].argmax()]
        out += w * F.cross_entropy(item2, t)
    return out

parser = argparse.ArgumentParser(description='Training DeepSleepNet')
parser.add_argument('--sampling', type=str, default='no',
                    help='Sampling methods: no, over, down')
parser.add_argument('--seq_len', type=int, default=25,
                    help='sequence length (default: 25)')
parser.add_argument('--pretrain_lr', type=float, default=10e-4,
                    help='learning rate of cnn')
parser.add_argument('--finetune_lr1', type=float, default=10e-6,
                    help='learning rate of lstm')
parser.add_argument('--finetune_lr2', type=float, default=10e-4)
parser.add_argument('--pretrain_epoch', type=int, default=100,
                    help='epoch number of cnn')
parser.add_argument('--pretrain_batch_size', type=int, default=100)
parser.add_argument('--finetune_epoch', type=int, default=200,
                    help='epoch number of lstm')
parser.add_argument('--finetune_batch_size', type=int, default=10)
parser.add_argument('--n_dataset', type=int, default=100000)
parser.add_argument('--pretrained_step1', type=bool, default=False)
parser.add_argument('--pretrained_step2', type=bool, default=False)
parser.add_argument('--save_sampled_data', type=bool, default=False)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--mode', type=str, default='train-test',
                    help="modes: 'train-test', 'train_only' , 'test_only'")
parser.add_argument('--weighted_CE', type=bool,default=False)
args = parser.parse_args()
interval = 1000
#######################################################
GPU_NUM = args.gpu
sampling = args.sampling
n_classes = 5
pretrain_epochs = args.pretrain_epoch
finetune_epochs = args.finetune_epoch
cnn_batch_size = args.pretrain_batch_size
seq_batch_size = args.finetune_batch_size
seq_length = args.seq_len
params = {'lr_c':args.pretrain_lr, 'lr_s1':args.finetune_lr1, 'lr_s2':args.finetune_lr2, 'wd':10e-3, 'betas':[0.9, 0.999]}
#######################################################
save_dataset = args.save_sampled_data
pre_trained_conv = args.pretrained_step1
pre_trained_seq = args.pretrained_step2
_psg = './Dataset/shhs_psg/'
_hyp = './Dataset/shhs_hyp/'
psg_filepath = _psg
hyp_filepath = _hyp
n_data_to_use = args.n_dataset
device = 'cuda:{}'.format(GPU_NUM) if torch.cuda.is_available() else 'cpu'
output_dir = './Output/Conv_weight/'
criterion = nn.CrossEntropyLoss().to(device)
#######################################
a = np.concatenate(hyp_data)
if args.mode == 'train-test' or args.mode == 'train_only':
    if not pre_trained_conv:
        if (not os.path.exists('train_data_{}sampling'.format(sampling))) or (not os.path.exists('valid_data_{}sampling'.format(sampling))):
            psg_filelist = os.listdir(psg_filepath)
            hyp_filelist = os.listdir(hyp_filepath)
            psg_data, hyp_data, psg_val, hyp_val = [], [], [], []
            for i in tqdm(range(len(psg_filelist[:n_data_to_use]))):
                psg_data.append(preprocess_data(psg_filepath, psg_filelist[i]))
                hyp_data.append(load_header(hyp_filepath, hyp_filelist[i]))
            train, val, test = train_data_split(psg_data, hyp_data)
            #train_dataset = dataset(*train)
            #valid_dataset = dataset(*val)
            #test_dataset = dataset(*test)

            if sampling == 'over':
                train_dataset.x, train_dataset.y = get_balance_class_oversample(train_dataset.x, train_dataset.y)
                print('Oversampling Done')

            elif sampling == 'down':
                train_dataset.x, train_dataset.y = get_balance_class_downsample(train_dataset.x, train_dataset.y)
                print('Downsampling Done')
            else:
                pass
            if save_dataset:
                torch.save(train_dataset, 'train_data_{}sampling'.format(sampling))
                torch.save(valid_dataset, 'valid_data_{}sampling'.format(sampling))
                torch.save(test_dataset, 'test_data_{}sampling'.format(sampling))
                print('Saved {}sampled dataset'.format(sampling))
        else:
            train_dataset = torch.load('train_data_{}sampling'.format(sampling))
            valid_dataset = torch.load('valid_data_{}sampling'.format(sampling))
            print('Loaded {}sampled dataset'.format(sampling))
        #print('Shape of training set', train_dataset.x.shape, train_dataset.y.shape)
        #print('Shape of validation set', valid_dataset.x.shape, valid_dataset.y.shape)

        train_dataset = CustomDataset(*train, True, True, 1)
        valid_dataset = CustomDataset(*val, False, True, 1)

        train_data_loader = DataLoader(train_dataset, batch_size=cnn_batch_size, shuffle=True)
        valid_data_loader = DataLoader(valid_dataset, batch_size=cnn_batch_size, shuffle=False)
        conv_model = DeepFeatureNet().to(device=device)
        optimizer_conv = optim.Adam(
            [{'params': conv_model.conv1d_s[0].parameters(), 'weight_decay': params['wd']},
             {'params': conv_model.conv1d_l[0].parameters(), 'weight_decay': params['wd']},
             {'params': conv_model.conv1d_s[1:].parameters()}, {'params': conv_model.conv1d_l[1:].parameters()},
             {'params': conv_model.fc.parameters()}], weight_decay=0.003, lr=params['lr_s1'])
    ############## Training ##############
    if not pre_trained_conv:
        max_valid_f1 = 0
        print('##################### * Training started ... ####################')
        for epoch in tqdm(range(pretrain_epochs)):
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            train_epoch_loss = 0.0
            valid_epoch_loss = 0.0
            train_pred = []
            valid_pred = []
            conv_model.train()
            for j, train in enumerate(train_data_loader):
                x, y = train
                optimizer_conv.zero_grad()
                output, _ = conv_model(x.to(device))
                if epoch >= pretrain_epochs // 3 and args.weighted_CE:
                    loss = weighted_loss(F.softmax(output,1), y.to(device=device, dtype=torch.long), cf_F1)
                else:
                    loss = criterion(output, y.to(device=device, dtype=torch.long))
                train_epoch_loss += loss
                loss.backward()
                optimizer_conv.step()
                pred_y_prob = F.softmax(output, dim=1)
                pred_y = torch.argmax(pred_y_prob, dim=1).detach().cpu().numpy()
                train_pred.append(pred_y)
            train_epoch_loss = train_epoch_loss / (j+1)
            train_pred = np.concatenate(train_pred)
            t_labels = train_data_loader.dataset.y.numpy()
            train_acc = int(sum(train_pred == t_labels)) / len(t_labels)
            train_f1 = f1_score(t_labels, train_pred, average='macro')
            t_cm = confusion_matrix(t_labels, train_pred)
            cf_F1 = []
            for ii in range(5):
                for jj in range(5):
                    cf_F1.append((2 * t_cm[ii][jj]) / (sum(t_cm[ii]) + sum(np.transpose(t_cm)[jj])))
            cf_F1 = torch.tensor(cf_F1).reshape([5, 5]).to(device)

            with torch.no_grad():
                conv_model.eval()
                for k, valid in enumerate(valid_data_loader):
                    x, y = valid
                    output, _ = conv_model(x.to(device))
                    if epoch >= pretrain_epochs // 3 and args.weighted_CE:
                        loss = weighted_loss(F.softmax(output,1), y.to(device=device, dtype=torch.long), v_cf_F1)
                    else:
                        loss = criterion(output, y.to(device=device, dtype=torch.long))
                    valid_epoch_loss += loss
                    pred_y_prob = F.softmax(output, dim=1)
                    pred_y = torch.argmax(pred_y_prob, dim=1).detach().cpu().numpy()
                    valid_pred.append(pred_y)
                valid_epoch_loss = valid_epoch_loss / (k+1)
                valid_pred = np.concatenate(valid_pred)
                v_labels = valid_data_loader.dataset.y.numpy()
                valid_acc = int(sum(valid_pred == v_labels)) / len(v_labels)
                valid_f1 = f1_score(v_labels, valid_pred, average='macro')
                v_cm = confusion_matrix(v_labels, valid_pred)
                v_cf_F1 = []
                for ii in range(5):
                    for jj in range(5):
                        v_cf_F1.append((2 * v_cm[ii][jj]) / (sum(v_cm[ii]) + sum(np.transpose(v_cm)[jj])))
                v_cf_F1 = torch.tensor(v_cf_F1).reshape([5, 5]).to(device)

                print('\n* [Epoch{}] [TRAIN] loss:{:.3f}, acc:{:.3f}, f1:{:.3f} // [VALID] loss:{:.3f}, acc:{:.3f}, f1:{:.3f}, max_valid_f1:{:.3f}'
                    .format(epoch + 1, train_epoch_loss, train_acc, train_f1, valid_epoch_loss, valid_acc, valid_f1, max_valid_f1))
                if valid_f1 > max_valid_f1:
                    print('** [Highest VALID f1] loss:{:.3f}, acc:{:.3f}, f1:{:.3f}'.format(valid_epoch_loss, valid_acc, valid_f1))
                    print('Valid Confusion Matrix\n', confusion_matrix(v_labels, valid_pred))
                    torch.save(conv_model.state_dict(), output_dir+'model_conv_{}sampling'.format(sampling))
                    print("* Saved model weights, Path:'{}model_conv_{}sampling'"
                          .format(os.getcwd().replace("\\",'/')+output_dir.replace('.',''), sampling))
                    max_valid_f1 = valid_f1

        print('#################### Pre-training finished ####################')
    else:
        output_dir = './Output/Conv_weight/'
        conv_model = DeepFeatureNet().to(device=device)
        conv_model.load_state_dict(torch.load(output_dir + 'model_conv_{}sampling'.format(sampling)))
        print("Loading saved weight from '{}model_conv_{}sampling'"
              .format(os.getcwd().replace("\\", '/') + output_dir.replace('.', ''), sampling))
        optimizer_conv = optim.Adam(
            [{'params': conv_model.conv1d_s[0].parameters(), 'weight_decay': params['wd']},
             {'params': conv_model.conv1d_l[0].parameters(), 'weight_decay': params['wd']},
             {'params': conv_model.conv1d_s[1:].parameters()}, {'params': conv_model.conv1d_l[1:].parameters()},
             {'params': conv_model.fc.parameters()}],weight_decay=0.003, lr=params['lr_s1'])

    ############## Seq-model Training ##############
    psg_filelist = os.listdir(psg_filepath)
    hyp_filelist = os.listdir(hyp_filepath)
    psg_data, hyp_data, psg_val, hyp_val = [], [], [], []
    for i in tqdm(range(len(psg_filelist[:n_data_to_use]))):
        psg_data.append(preprocess_data(psg_filepath, psg_filelist[i]))
        hyp_data.append(load_header(hyp_filepath, hyp_filelist[i]))
    train, val, test = train_data_split(psg_data, hyp_data)

    train_dataset = dataset(*train)
    if not os.path.exists('valid_data_{}sampling'.format(sampling)):
        valid_dataset = dataset(*val)
        test_dataset = dataset(*test)
        if save_dataset:
            torch.save(valid_dataset, 'valid_data_{}sampling'.format(sampling))
            torch.save(test_dataset, 'test_data_{}sampling'.format(sampling))
            print('Saved {}sampled dataset'.format(sampling))
    else:
        valid_dataset = torch.load('valid_data_{}sampling'.format(sampling))
        print('Loaded {}sampled dataset'.format(sampling))

    s = seq_length
    b, c, l = train_dataset.x.shape
    train_dataset.x, train_dataset.y = train_dataset.x[:b-b%s].view(-1, s, c, l), train_dataset.y[:b-b%s].view(-1, s)
    b, c, l = valid_dataset.x.shape
    valid_dataset.x, valid_dataset.y = valid_dataset.x[:b-b%s].view(-1, s, c, l), valid_dataset.y[:b-b%s].view(-1, s)
    print('Loaded {}sampled dataset'.format(sampling))
    print('Shape of training set', train_dataset.x.shape, train_dataset.y.shape)
    print('Shape of validation set', valid_dataset.x.shape, valid_dataset.y.shape)

    train_dataset = CustomDataset(*train, True, False, 25)
    valid_dataset = CustomDataset(*val, False, False, 25)

    train_sampler = torch.utils.data.SequentialSampler(train_dataset)
    val_sampler = torch.utils.data.SequentialSampler(valid_dataset)
    train_data_loader = DataLoader(train_dataset, batch_size=seq_batch_size, shuffle=False, sampler=train_sampler)
    valid_data_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, sampler=val_sampler)

    seq_model = DeepSleepNet(bidirectional=True).to(device=device)
    #torch.nn.utils.clip_grad_norm_(seq_model.parameters(), 10)
    optimizer_seq = optim.Adam(seq_model.parameters(), weight_decay=0.003, lr=params['lr_s2'])
    criterion = nn.CrossEntropyLoss()
    output_dir = './Output/Seq_weight/'

    print('##################### * Fine-tuning Training started ... ####################')
    print('* Training dataset: {} // Validation dataset: {}'.format(len(train_data_loader.dataset.x), len(valid_data_loader.dataset.x)))
    bs = seq_length*seq_batch_size
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    max_valid_f1 = 0
    if not pre_trained_seq:
        for epoch in tqdm(range(finetune_epochs)):
            train_epoch_loss = 0
            valid_epoch_loss = 0
            train_pred = []
            valid_pred = []
            conv_model.train()
            seq_model.train()
            for j, train in enumerate(train_data_loader):
                x, y = train
                b, s, c, l = x.shape
                optimizer_conv.zero_grad()
                optimizer_seq.zero_grad()
                _, x = conv_model(x.view(b*s, c,-1).to(device))
                hidden, cell = seq_model.init_state(b)
                output = seq_model(x.view(b, s, -1), hidden.to(device), cell.to(device))
                if epoch >= pretrain_epochs // 3 and args.weighted_CE:
                    loss = weighted_loss(F.softmax(output,1).view(-1, 5), y.view(-1).to(device=device, dtype=torch.long), cf_F1)
                else:
                    loss = criterion(output.view(-1, 5), y.view(-1).to(device=device, dtype=torch.long))
                train_epoch_loss += loss
                loss.backward()
                optimizer_conv.step()
                optimizer_seq.step()
                pred_y_prob = F.softmax(output.view(-1, 5), dim=1)
                pred_y = torch.argmax(pred_y_prob, dim=1).detach().cpu().numpy()
                train_pred.append(pred_y)
                if j+1 % interval == 0:
                    with torch.no_grad():
                        conv_model.eval()
                        seq_model.eval()
                        for k, valid in enumerate(valid_data_loader):
                            x, y = valid
                            b, s, c, l = x.shape
                            _, x = conv_model(x.view(b * s, c, -1).to(device))
                            hidden, cell = seq_model.init_state(b)
                            output = seq_model(x.view(b, s, -1), hidden.to(device), cell.to(device))
                            if epoch >= pretrain_epochs // 3 and args.weighted_CE:
                                loss = weighted_loss(F.softmax(output,1).view(-1, 5), y.view(-1).to(device=device, dtype=torch.long), v_cf_F1)
                            else:
                                loss = criterion(output.view(-1, 5), y.view(-1).to(device=device, dtype=torch.long))
                            valid_epoch_loss += loss
                            pred_y_prob = F.softmax(output.view(-1, 5), dim=1)
                            pred_y = torch.argmax(pred_y_prob, dim=1).detach().cpu().numpy()
                            valid_pred.append(pred_y)
                    valid_epoch_loss = valid_epoch_loss / (k + 1)
                    valid_pred = np.concatenate(valid_pred)
                    v_labels = valid_data_loader.dataset.y.numpy().flatten()
                    valid_acc = int(sum(valid_pred == v_labels)) / len(v_labels)
                    valid_f1 = f1_score(v_labels, valid_pred, average='macro')
                    v_cm = confusion_matrix(v_labels, valid_pred)
                    v_cf_F1 = []
                    for ii in range(5):
                        for jj in range(5):
                            v_cf_F1.append((2 * v_cm[ii][jj]) / (sum(v_cm[ii]) + sum(np.transpose(v_cm)[jj])))
                    v_cf_F1 = torch.tensor(v_cf_F1).reshape([5, 5]).to(device)

            train_epoch_loss = train_epoch_loss / (j+1)
            train_pred = np.concatenate(train_pred)
            t_labels = train_data_loader.dataset.y.numpy().flatten()
            train_acc = int(sum(train_pred == t_labels)) / len(t_labels)
            train_f1 = f1_score(t_labels, train_pred, average='macro')
            t_cm = confusion_matrix(t_labels, train_pred)
            cf_F1 = []
            for ii in range(5):
                for jj in range(5):
                    cf_F1.append((2 * t_cm[ii][jj]) / (sum(t_cm[ii]) + sum(np.transpose(t_cm)[jj])))
            cf_F1 = torch.tensor(cf_F1).reshape([5, 5]).to(device)


            print('\n* [Epoch{}] [TRAIN] loss:{:.3f}, acc:{:.3f}, f1:{:.3f} // [VALID] loss:{:.3f}, acc:{:.3f}, f1:{:.3f}, max_valid_f1:{:.3f}'
                .format(epoch + 1, train_epoch_loss, train_acc, train_f1, valid_epoch_loss, valid_acc, valid_f1, max_valid_f1))
            if valid_f1 > max_valid_f1:
                print('** [Highest VALID f1] loss:{:.3f}, acc:{:.3f}, f1:{:.3f}'.format(valid_epoch_loss, valid_acc, valid_f1))
                print('Valid Confusion Matrix\n', confusion_matrix(v_labels, valid_pred))
                torch.save(seq_model.state_dict(), output_dir+'model_seq_{}sampling'.format(sampling))
                print("* Saved model weights, Path:'{}model_seq_{}sampling"
                      .format(os.getcwd().replace("\\",'/')+output_dir.replace('.',''), sampling))
                max_valid_f1 = valid_f1

        print('#################### Fine-tuning finished ####################')
    else:
        seq_model.load_state_dict(torch.load(output_dir + 'model_seq_{}sampling'.format(sampling)))
        print("Loading saved weight from '{}model_seq_{}sampling'".format(
            os.getcwd().replace("\\", '/') + output_dir.replace('.', ''), sampling))
        seq_model.eval()

if args.mode == 'train-test' or args.mode == 'test_only':
    print('Prediction stated')
    if not os.path.exists('test_data_{}sampling'.format(sampling)):
        psg_filelist = os.listdir(psg_filepath)
        hyp_filelist = os.listdir(hyp_filepath)
        psg_data, hyp_data, psg_val, hyp_val = [], [], [], []
        for i in tqdm(range(len(psg_filelist[:n_data_to_use]))):
            psg_data.append(preprocess_data(psg_filepath, psg_filelist[i]))
            hyp_data.append(load_header(hyp_filepath, hyp_filelist[i]))

        train, val, test = train_data_split(psg_data, hyp_data)
        test_dataset = dataset(*test)
        torch.save(test_dataset, 'test_data_{}sampling'.format(sampling))
    else:
        test_dataset = torch.load('test_data_{}sampling'.format(sampling))

    criterion = nn.CrossEntropyLoss()
    test_pred = []
    output_dir = './Output/Conv_weight/'
    conv_model = DeepFeatureNet().to(device=device)
    conv_model.load_state_dict(torch.load(output_dir + 'model_conv_{}sampling'.format(sampling)))
    print("Loading saved weight from '{}model_conv_{}sampling'"
          .format(os.getcwd().replace("\\", '/') + output_dir.replace('.', ''), sampling))
    output_dir = './Output/Seq_weight/'
    seq_model = DeepSleepNet(bidirectional=True).to(device=device)
    seq_model.load_state_dict(torch.load(output_dir + 'model_seq_{}sampling'.format(sampling)))
    print("Loading saved weight from '{}model_seq_{}sampling'"
          .format(os.getcwd().replace("\\", '/') + output_dir.replace('.', ''), sampling))

    s = seq_length
    b, c, l = test_dataset.x.shape
    test_dataset.x, test_dataset.y = test_dataset.x[:b-b%s].view(-1, s, c, l), test_dataset.y[:b-b%s].view(-1, s)

    print('Shape of test set', test_dataset.x.shape, test_dataset.y.shape)
    test_data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    conv_model.eval()
    seq_model.eval()
    with torch.no_grad():
        for i, test in enumerate(test_data_loader):
            x, y = test
            b, s, c, l = x.shape
            _, x = conv_model(x.view(b * s, c, -1).to(device))
            hidden, cell = seq_model.init_state(b)
            output = seq_model(x.view(b, s, -1), hidden.to(device), cell.to(device))
            pred_y_prob = F.softmax(output.view(-1, 5), dim=1)
            pred_y = torch.argmax(pred_y_prob, dim=1).detach().cpu().numpy()
            test_pred.append(pred_y)

    test_pred = np.concatenate(test_pred)
    labels = test_data_loader.dataset.y.numpy().flatten()
    test_acc = int(sum(test_pred == labels)) / len(labels)
    test_f1 = f1_score(labels, test_pred, average='macro')
    print('\n* [Test] acc:{:.3f}, f1:{:.3f}'.format(test_acc, test_f1))
    print('Test Confusion Matrix\n', confusion_matrix(labels, test_pred))
    print('#################### Prediction finished ####################')

