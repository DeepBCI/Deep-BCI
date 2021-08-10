import numpy as np
import sys
import os
import logging
from gigadataset import GigaDataset
import torch
import random

def param_size(model):
    n_params = sum(
        np.prod(v.size()) for k, v in model.named_parameters() if not k.startswith('aux_head'))
    return n_params / 1024. / 1024.

def blockPrint():
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    sys.stdout = sys.__stdout__

def get_data_subject_dependent_single_3cls_1s(args,fold_idx,opt=3):
    print(f"target subject : {fold_idx}")
    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    x_data = np.load(args.data_root + '/x_data_3cls.npy', mmap_mode='r')
    y_data = np.load(args.data_root + '/y_data_3cls.npy')
    if opt ==1:#rest vs hand
        y_data = np.where(y_data==2,1,0)
    elif opt == 2:
        x_data = x_data[np.where(y_data!=2)]
        y_data = y_data[np.where(y_data!=2)]
    elif opt ==3:
        y_data = y_data
    else:
        raise

    x_data = np.expand_dims(x_data,axis=1)
    in_chans = x_data.shape[2]
    input_time_length = x_data.shape[3]

    x_data = x_data.reshape(108, -1, in_chans, input_time_length)
    y_data = y_data.reshape(108, -1)

    args.n_class = len(np.unique(y_data))
    args.n_ch = 20
    args.n_time = 250

    train_set_list = []
    valid_set_list = []
    for s_id in range(0, 54):
        train_len = int(0.9 * 400)
        valid_len = 400 - train_len
        temp_dataset= GigaDataset([np.expand_dims(x_data[s_id, :, :, :], axis=1), y_data[s_id, :]], s_id)
        train_set_temp, valid_set_temp = torch.utils.data.dataset.random_split(temp_dataset, [train_len, valid_len])
        train_set_list.append(train_set_temp)
        valid_set_list.append(valid_set_temp)

    for s_id in range(54, 108):
        train_len = int(0.9 * 200)
        valid_len = 200 - train_len
        temp_dataset= GigaDataset([np.expand_dims(x_data[s_id, list(range(0, 100)) + list(range(200, 300)), :, :]
                                                  , axis=1), y_data[s_id, list(range(0, 100)) + list(range(200, 300))]], s_id)
        train_set_temp, valid_set_temp = torch.utils.data.dataset.random_split(temp_dataset, [train_len, valid_len])
        train_set_list.append(train_set_temp)
        valid_set_list.append(valid_set_temp)

    test_set_list = []
    for s_id in range(54, 108):
        test_set_list.append(GigaDataset([np.expand_dims(x_data[s_id, list(range(100, 200)) + list(range(300, 400)), :, :], axis=1),
                                           y_data[s_id, list(range(100, 200)) + list(range(300, 400))]], s_id))

    train_set = torch.utils.data.ConcatDataset( [train_set_list[fold_idx], train_set_list[fold_idx+54]])
    valid_set = torch.utils.data.ConcatDataset( [valid_set_list[fold_idx], valid_set_list[fold_idx+54]])
    test_set = test_set_list[fold_idx]
    return train_set, valid_set, test_set, args

