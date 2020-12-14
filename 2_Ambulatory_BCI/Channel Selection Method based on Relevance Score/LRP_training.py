import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import scipy.io as scio
#matplotlib inline
import os

import torch

import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
#from torchvision import datasets, transforms

from tool_investigate_model import InnvestigateModel
from tool_utils import Flatten

from LRP_model import Net
from LRP_data_preprocessing import EEG_val_dataset, EEG_train_dataset


import seaborn as sns

# Network parameters
class Params(object):
    batch_size = 10
    test_batch_size = 10
    epochs = 85
    lr = 0.01
    momentum = 0.5
    no_cuda = False
    seed = 1
    log_interval = 10
    
    def __init__(self):
        pass

args = Params()
torch.manual_seed(args.seed)
device = torch.device("cuda")
kwargs = {}

model = Net().double()
model.load_state_dict(torch.load("model_parameter_ch118/stft_al/cv_al_3"))

# Convert to innvestigate model
inn_model = InnvestigateModel(model, lrp_exponent=2,
                              method="e-rule",
                              beta=.5)

#load data, dataloader
val_dataset = EEG_val_dataset()
val_loader=DataLoader(dataset=val_dataset, batch_size=args.test_batch_size, shuffle=True, num_workers=0)

train_dataset = EEG_train_dataset()
train_loader=DataLoader(dataset=train_dataset, batch_size=args.test_batch_size, shuffle=True, num_workers=0)



# empty array set (for concate the relevance vector)
left_relevance_score = np.zeros(11800000).reshape(10,118,200,50)
right_relevance_score = np.zeros(11800000).reshape(10,118,200,50)
    
left_relevance_score = torch.from_numpy(left_relevance_score)
right_relevance_score = torch.from_numpy(right_relevance_score)


# Drawing heatmap
def Draw_heatmap(left_input, right_input):
    
    # make empty vector to get accumulated vector of trials
    trial_sum_left = np.zeros(1180000).reshape(1,118,200,50)
    trial_sum_right = np.zeros(1180000).reshape(1,118,200,50)
    
    trial_sum_left = torch.from_numpy(trial_sum_left)
    trial_sum_right = torch.from_numpy(trial_sum_right)

    print("size",left_input.size(0))

    # Sum(Trial)
    for idx in range(left_input.size(0)):

        trial_sum_left[0,:,:,:] += left_input[idx,:,:,:]
        trial_sum_right[0,:,:,:] += right_input[idx, :,:,:]

    
    print("trial_left", trial_sum_left.shape)
    print("trial_right", trial_sum_right.shape)

    left = trial_sum_left.numpy()
    right = trial_sum_right.numpy()
 
    #scio.savemat("relevancescore_ch118/stft_aa/left_1.mat", {"left" : left})
    #scio.savemat("relevancescore_ch118/stft_aa/right_1.mat", {"right" : right})
    
    


# Calculate rank of channels
def Calculate_rank(left_input, right_input):
    
    # left & right input_shape = [ 262, 118, 200, 50]

    cal_trial_sum_left = np.zeros(1180000).reshape(1,118,200,50)
    cal_trial_sum_right = np.zeros(1180000).reshape(1,118,200,50)
    
    cal_trial_sum_left = torch.from_numpy(cal_trial_sum_left)
    cal_trial_sum_right = torch.from_numpy(cal_trial_sum_right)

    
    time_seq_len = left_input.size(3)
    freq_len = left_input.size(2)
    

    # Sum(Trial)
    for idx in range(left_input.size(0)):
        cal_trial_sum_left[0,:,:,:] += left_input[idx,:,:,:]
        cal_trial_sum_right[0,:,:,:] += right_input[idx, :,:,:]

    print("len_val", len(val_dataset))

    cal_trial_sum_left = cal_trial_sum_left.squeeze() / len(val_dataset)
    cal_trial_sum_right = cal_trial_sum_right.squeeze() / len(val_dataset)
    

    cal_time_freq_sum_left = np.zeros(118).reshape(118,1)
    cal_time_freq_sum_right = np.zeros(118).reshape(118,1)
    
    cal_time_freq_sum_left = torch.from_numpy(cal_time_freq_sum_left)
    cal_time_freq_sum_right = torch.from_numpy(cal_time_freq_sum_right)


    # Sum(Time and freq domain)

    for time_idx in range(time_seq_len):
        
        for freq_idx in range(freq_len):
        
            cal_time_freq_sum_left[:,0] += cal_trial_sum_left[:,freq_idx,time_idx]
            cal_time_freq_sum_right[:,0] += cal_trial_sum_right[:,freq_idx,time_idx]
    

    #cal_time_freq_sum_left = cal_time_freq_sum_left / time_seq_len
    #cal_time_freq_sum_right = cal_time_freq_sum_right / time_seq_len


    print("===========")
    print(" plz  index that shows on cmd must be index+1 // matlab - python indexs are different")
    print("left signal's channels", sorted(range(len(cal_time_freq_sum_left)), key = lambda k: cal_time_freq_sum_left[k], reverse=True ) )
    print("right signal's channels", sorted(range(len(cal_time_freq_sum_right)), key = lambda k: cal_time_freq_sum_right[k], reverse=True ) )
    



for data, target in train_loader:
    
    target = target.long()

    batch_size = int(data.size()[0])
    
    evidence_for_class = []

    model_prediction, true_relevance = inn_model.innvestigate(in_tensor=data)
    
    
    
    for i in range(2):
        # Unfortunately, we had some issue with freeing pytorch memory, therefore 
        # we need to reevaluate the model separately for every class.



        model_prediction, input_relevance_values = inn_model.innvestigate(in_tensor=data, rel_for_class=i)


        # case left
        if i == 0:
            left_relevance_score = torch.cat([left_relevance_score,input_relevance_values], axis=0)
           
        # case right
        else:
            right_relevance_score = torch.cat([right_relevance_score,input_relevance_values], axis=0)



#Draw_heatmap(left_relevance_score, right_relevance_score)
Calculate_rank(left_relevance_score,right_relevance_score)
