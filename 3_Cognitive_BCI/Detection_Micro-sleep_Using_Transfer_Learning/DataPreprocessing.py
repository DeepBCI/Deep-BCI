import os
import torch
import numpy as np
from tqdm import tqdm
from scipy import io

def load_data(type):
    list = os.listdir(type)
    data_all = {'len':len(list)}
    label_all = {'len':len(list)}
    for i, file in tqdm(enumerate(list), desc='Data Load'):
        f = io.loadmat(type+'\\'+file)
        data = f['epo']['x']
        label = f['epo']['y']

        data_all[i] = data[0][0]
        label_all[i] = label[0][0]
    return (data_all, label_all)

def load_single_data(results_dir, type, test_sub):
    if test_sub >= 9:
        f = io.loadmat(results_dir + type + str(test_sub + 1))
    else:
        f = io.loadmat(results_dir + type + '0' + str(test_sub + 1))
    data = f['epo']['x']
    label = f['epo']['y']
    data = data[0][0]
    label = label[0][0]

    return (data, label)

def train_val_split(data, label, test_sub, sample_location=0):

    trainx = []
    trainy = []
    for sub in range(data['len']):
        if test_sub == sub:
            valx = data[sub]
            valy = label[sub]
        else:
            trainx += [data[sub]]
            trainy += [label[sub]]

    trainx = np.concatenate(trainx, sample_location)
    trainy = np.concatenate(trainy, 1)

    return (trainx, trainy, valx, valy)


def gpu_setting(gpu_num):
    txt = 'GPU available!' if torch.cuda.is_available() else 'GPU can not find'
    device = torch.device(f'cuda:{gpu_num}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(gpu_num))
        print('Allocated:', round(torch.cuda.memory_allocated(gpu_num) / 1024 ** 3, 1), 'GB')
        print('Cached:', round(torch.cuda.memory_reserved(gpu_num) / 1024 ** 3, 1), 'GB')
    print(txt)

    n_gpu = torch.cuda.current_device()
    torch.cuda.device(n_gpu)
    cuda = torch.device('cuda')
    return cuda