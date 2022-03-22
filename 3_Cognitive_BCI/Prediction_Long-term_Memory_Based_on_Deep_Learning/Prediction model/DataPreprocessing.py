import torch
import numpy as np
import scipy.io as sio

def gpu_setting(gpu_num):
    # Setting cuda for using gpu
    txt = 'GPU available!' if torch.cuda.is_available() else 'GPU can not find'
    device = torch.device(f'cuda:{gpu_num}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(gpu_num))
        print('Allocated:', round(torch.cuda.memory_allocated(gpu_num) / 1024 ** 3, 1), 'GB')
        print('Cached:', round(torch.cuda.memory_cached(gpu_num) / 1024 ** 3, 1), 'GB')
    print(txt)

    n_gpu = torch.cuda.current_device()
    torch.cuda.device(n_gpu)
    cuda = torch.device('cuda')
    return cuda

def split_train_test(x, y, sub):

    trainx = []
    trainy = []
    testx = []
    testy = []
    for i in range(len(x)):
        if i == sub:
            testx = x[i]
            testy = y[i]
        else:
            trainx += [x[i]]
            trainy += [y[i]]
    # CNN
    # trainx = np.concatenate(trainx, axis=3)
    # trainy = np.concatenate(trainy)

    # MLP
    trainx = np.concatenate(trainx, axis=2)
    trainy = np.concatenate(trainy, axis=0)

    temp_X = np.zeros([trainx.shape[0]*trainx.shape[1], trainx.shape[2]])
    for i in range(trainx.shape[1]):
        temp = trainx[:, i, :]
        temp = np.squeeze(temp)
        temp_X[trainx.shape[0]*i:trainx.shape[0]*(i+1), :] = temp
    trainx = temp_X

    temp_X = np.zeros([testx.shape[0]*testx.shape[1], testx.shape[2]])
    for i in range(testx.shape[1]):
        temp = testx[:, i, :]
        temp = np.squeeze(temp)
        temp_X[testx.shape[0]*i:testx.shape[0]*(i+1), :] = temp
    testx = temp_X

    return (trainx, trainy, testx, testy)

# EEG data load (Before nap)
def preprocess_data(datapath, filename):
    f = sio.loadmat(datapath + filename)
    out = f.get('x')
    return out

# True label load (After nap)
def load_header(datapath, filename):
    f = sio.loadmat(datapath + filename)
    out = f.get('y')
    return out
