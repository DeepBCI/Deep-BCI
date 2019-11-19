
from sklearn.feature_selection import mutual_info_classif as MINF
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import numpy as np
import scipy
import scipy.io
import fnmatch
import os
from scipy import signal
import matplotlib.pyplot as plt
import time
import random




# since 2018 // the data contains all the behavioral labels. (you should code the methods to label the data)
def ext_files(direc_cond):
    # direc_cond refers to the directory and prefix
    lom = [] # list of matrix.
    for file in os.listdir(direc_cond):
        if fnmatch.fnmatch(file, '*.mat'):
            lom.append(file)
    lom.sort()
    return lom

def ext_epochs(direc_cond, lom, indexforlom):
    # first, you have to use ext_files to extract list of files. => lom : list of mat-file
    if lom[0] == 'pjs':
        return scipy.io.loadmat(direc_cond+'/dataset{}_parsed.mat'.format(indexforlom))['ep']

    lom.sort()
    epochs = scipy.io.loadmat(direc_cond +'/'+ lom[indexforlom] )['eeg']['epoch'][0,0]
    return epochs


def ext_label_tot(direc_cond, lom, indexforlom ,label_cond):
    # sta memnas the state you are interested in. (1 or 2)
    # label_cond means the condition of labeling data. e.g. "maxrel" means that you want to label eeg data with maxinvfano in the sbj.regressor
    # if you want to change this label_cond make sure that you coded in the dataLoader.ext_label
    # "maxrel" : maximum reliability (wherther that maximum reliabiltiy is came from MB or MF
    # "goal"   : goal condition (specific versus flexible
    # "Pmb5"   : label trials with pmb exceed 0.5 as mb and label as mf for others.
    # "action" : label for right and left action chosen
    behav = scipy.io.loadmat(direc_cond +'/'+ lom[indexforlom] )['eeg']['sbj']

    # new dataset will not have separate folders for each strategies
    # this part seems to be unnecessary, so I commented them out
    # if '/sta1' in direc_cond:
    #     locking = 1
    # elif '/sta2' in direc_cond:
    #     locking = 2
    # else:
    #     print("ERROR!!!!???? : CHECK LOCKING EVENT? E.G. STA1?")

    if label_cond.lower() == 'maxrel':
        reg = scipy.io.loadmat(direc_cond +'/'+ lom[indexforlom] )['eeg']['reg']

        # m1 = reg[0, 0][0, 24][0, 0][1][6,:]  # MB
        # m2 = reg[0, 0][0, 25][0, 0][1][6,:]  # MF
        # mm = reg[0, 0][0, 21][0, 0][1][6,:]  # used to be indexed 1, updated data has length 1
        m1 = reg[0, 0][0, 24][0, 0][0][6,:]  # MB
        m2 = reg[0, 0][0, 25][0, 0][0][6,:]  # MF
        mm = reg[0, 0][0, 21][0, 0][0][6,:]

        trials = int(m1.shape[0]/3)
        labels = np.zeros((trials * 2,)) + 2

        indices = np.int16(np.linspace(0,trials-1,trials)*3 + 1)  # action 1
        m1_real = m1[indices]
        m2_real = m2[indices]
        mm_real = mm[indices]
        m1_indice = np.where(mm_real == m1_real)[0]
        m2_indice = np.where(mm_real == m2_real)[0]

        labels[m1_indice * 2] = 0
        labels[m2_indice * 2] = 1

        indices = np.int16(np.linspace(0,trials-1,trials)*3 + 2)  # action 2
        m1_real = m1[indices]
        m2_real = m2[indices]
        mm_real = mm[indices]
        m1_indice = np.where(mm_real == m1_real)[0]
        m2_indice = np.where(mm_real == m2_real)[0]

        labels[m1_indice * 2 + 1] = 0
        labels[m2_indice * 2 + 1] = 1
        regressors = []
        for iii in range(reg[0,0].shape[1]):
            labels = np.zeros((trials * 2,)) + 2
            labels[m1_indice * 2]=reg[0, 0][0,iii][0,0][0][6,m1_indice * 2]
            labels[m2_indice * 2]=reg[0, 0][0,iii][0,0][0][6,m2_indice * 2]
            labels[m1_indice * 2 + 1]=reg[0, 0][0,iii][0,0][0][6,m1_indice * 2 + 1]
            labels[m2_indice * 2 + 1]=reg[0, 0][0,iii][0,0][0][6,m2_indice * 2 + 1]
            regressors.append(labels)


        # m1_set = set(m1_indice)
        # m2_set = set(m2_indice)
        # overlap = m1_set.intersection(m2_set)
        #
        # m1_indice = list(m1_set.difference(overlap))
        # m2_indice = list(m2_set.difference(overlap))
        #
        # m1_indice.sort()
        # m2_indice.sort()
        #
        # for i in range(len(m1_indice)):
        #     m1_indice[i] = m1_indice[i]
        # for i in range(len(m2_indice)):
        #     m2_indice[i] = m2_indice[i]

    # elif label_cond.lower() == 'goal':
    # elif label_cond.lower() == 'pmb5':

    elif label_cond.lower() == 'action':
        # not fixed yet
        sbj = scipy.io.loadmat(direc_cond + '/' + lom[indexforlom])['eeg']['sbj']
        action_hist1 = sbj[0, 0][:, 5 + 1]
        action_hist2 = sbj[0, 0][:, 5 + 2]
        trials = int(action_hist1.shape[0])
        m1_indice = np.where(action_hist1 == 1)
        m2_indice = np.where(action_hist2 == 2)

    else:
        # whatelse?
        print("ERROR!!!!???? : CHECK YOUR LABEL FROM '''PARAMETERS_EEG.JSON''' FILE")

    # now, return value will have sequence of numbers that are from either action1 or action2
    # i.e. labels = [act1, act2, act1, act2, act1, act2, ...]

    return regressors


def ext_label(direc_cond, lom, indexforlom ,label_cond):
    # sta memnas the state you are interested in. (1 or 2)
    # label_cond means the condition of labeling data. e.g. "maxrel" means that you want to label eeg data with maxinvfano in the sbj.regressor
    # if you want to change this label_cond make sure that you coded in the dataLoader.ext_label
    # "maxrel" : maximum reliability (wherther that maximum reliabiltiy is came from MB or MF
    # "goal"   : goal condition (specific versus flexible
    # "Pmb5"   : label trials with pmb exceed 0.5 as mb and label as mf for others.
    # "action" : label for right and left action chosen
    # behav = scipy.io.loadmat(direc_cond +'/'+ lom[indexforlom] )['eeg']['sbj']

    # new dataset will not have separate folders for each strategies
    # this part seems to be unnecessary, so I commented them out
    # if '/sta1' in direc_cond:
    #     locking = 1
    # elif '/sta2' in direc_cond:
    #     locking = 2
    # else:
    #     print("ERROR!!!!???? : CHECK LOCKING EVENT? E.G. STA1?")

    if lom[0] == 'pjs':
        return scipy.io.loadmat(direc_cond + '/dataset{}_parsed.mat'.format(indexforlom))['lb'][0, :]

    elif label_cond.lower() == 'maxrel':
        reg = scipy.io.loadmat(direc_cond +'/'+ lom[indexforlom] )['eeg']['reg']

        # m1 = reg[0, 0][0, 24][0, 0][1][6,:]  # MB
        # m2 = reg[0, 0][0, 25][0, 0][1][6,:]  # MF
        # mm = reg[0, 0][0, 21][0, 0][1][6,:]  # used to be indexed 1, updated data has length 1
        m1 = reg[0, 0][0, 24][0, 0][0][6,:]  # MB
        m2 = reg[0, 0][0, 25][0, 0][0][6,:]  # MF
        mm = reg[0, 0][0, 21][0, 0][0][6,:]

        trials = int(m1.shape[0]/3)
        labels = np.zeros((trials * 2,)) + 2

        indices = np.int16(np.linspace(0,trials-1,trials)*3 + 1)  # action 1
        m1_real = m1[indices]
        m2_real = m2[indices]
        mm_real = mm[indices]
        m1_indice = np.where(mm_real == m1_real)[0]
        m2_indice = np.where(mm_real == m2_real)[0]

        labels[m1_indice * 2] = 0
        labels[m2_indice * 2] = 1

        indices = np.int16(np.linspace(0,trials-1,trials)*3 + 2)  # action 2
        m1_real = m1[indices]
        m2_real = m2[indices]
        mm_real = mm[indices]
        m1_indice = np.where(mm_real == m1_real)[0]
        m2_indice = np.where(mm_real == m2_real)[0]

        labels[m1_indice * 2 + 1] = 0
        labels[m2_indice * 2 + 1] = 1

        # m1_set = set(m1_indice)
        # m2_set = set(m2_indice)
        # overlap = m1_set.intersection(m2_set)
        #
        # m1_indice = list(m1_set.difference(overlap))
        # m2_indice = list(m2_set.difference(overlap))
        #
        # m1_indice.sort()
        # m2_indice.sort()
        #
        # for i in range(len(m1_indice)):
        #     m1_indice[i] = m1_indice[i]
        # for i in range(len(m2_indice)):
        #     m2_indice[i] = m2_indice[i]

    # elif label_cond.lower() == 'goal':
    # elif label_cond.lower() == 'pmb5':

    elif label_cond.lower() == 'action':
        # not fixed yet
        sbj = scipy.io.loadmat(direc_cond + '/' + lom[indexforlom])['eeg']['sbj']
        action_hist1 = sbj[0, 0][:, 5 + 1]
        action_hist2 = sbj[0, 0][:, 5 + 2]
        trials = int(action_hist1.shape[0])
        labels = []
        for i in range(len(action_hist1)):
            labels.append(int(action_hist1[i]-1))
            labels.append(int(action_hist2[i]-1))
        labels = np.array(labels)
    else:
        # whatelse?
        print("ERROR!!!!???? : CHECK YOUR LABEL FROM '''PARAMETERS_EEG.JSON''' FILE")

    # now, return value will have sequence of numbers that are from either action1 or action2
    # i.e. labels = [act1, act2, act1, act2, act1, act2, ...]

    return labels

def parse_eplb(labels, epochs):
    MB_list = []
    MF_list = []

    # first, extract indexes where subject used Model Based strategy
    for i in range(labels.shape[0]):
        if labels[i] == 1:
            MB_list.append(i)
        if labels[i] == 0:
            MF_list.append(i)

    # extract MB, MF epoch and shuffle
    random.shuffle(MB_list)
    random.shuffle(MF_list)
    MB_epoch = epochs[:, :, MB_list]
    MF_epoch = epochs[:, :, MF_list]

    # cut by appropriate size
    index = min(len(MB_list), len(MF_list))
    MB_list = [1] * index
    MF_list = [0] * index
    MB_epoch = MB_epoch[:,:,:index]
    MF_epoch = MF_epoch[:,:,:index]
    ret_lb = np.concatenate((MB_list, MF_list), axis=0)
    ret_ep = np.concatenate((MB_epoch,MF_epoch), axis=2)
    return ret_lb, ret_ep


def parse_eplb_3(labels, epochs, inp):
    MB_list = []
    MF_list = []

    # first, extract indexes where subject used Model Based strategy
    for i in range(labels.shape[0]):
        if labels[i] == 1:
            MB_list.append(i)
        if labels[i] == 0:
            MF_list.append(i)
    # extract MB, MF epoch and shuffle
    MB_epoch = epochs[:,:,MB_list]
    MF_epoch = epochs[:,:,MF_list]
    # random.shuffle(MB_epoch)
    # random.shuffle(MF_epoch)
    MB_inp = inp[MB_list,:]
    MF_inp = inp[MF_list,:]
    # random.shuffle(MB_inp)
    # random.shuffle(MF_inp)

    # cut by appropriate size
    index = min(len(MB_list), len(MF_list))
    MB_list = [1] * index
    MF_list = [0] * index
    MB_epoch = MB_epoch[:,:,:index]
    MF_epoch = MF_epoch[:,:,:index]

    MB_inp = MB_inp[:index,:]
    MF_inp = MF_inp[:index,:]

    ret_lb = np.concatenate((MB_list, MF_list), axis=0)
    ret_ep = np.concatenate((MB_epoch,MF_epoch), axis=2)
    ret_inp = np.concatenate((MB_inp,MF_inp), axis=0)
    return ret_lb, ret_ep, ret_inp




def parse_eplb_after(labels, xtt):
    MB_list = []
    MF_list = []

    # first, extract indexes where subject used Model Based strategy
    for i in range(labels.shape[0]):
        if labels[i] == 1:
            MB_list.append(i)
        if labels[i] == 0:
            MF_list.append(i)
    # extract MB, MF epoch and shuffle
    MB_epoch = xtt[MB_list,:,:]
    MF_epoch = xtt[MF_list,:,:]
    # random.shuffle(MB_epoch)
    # random.shuffle(MF_epoch)
    # random.shuffle(MB_inp)
    # random.shuffle(MF_inp)

    # cut by appropriate size
    index = min(len(MB_list), len(MF_list))
    MB_list = [1] * index
    MF_list = [0] * index
    MB_epoch = MB_epoch[:index,:,:]
    MF_epoch = MF_epoch[:index,:,:]


    ret_lb = np.concatenate((MB_list, MF_list), axis=0)
    ret_ep = np.concatenate((MB_epoch,MF_epoch), axis=0)
    return ret_lb, ret_ep




def ext_input_lstm(direc_cond, lom, indexforlom, maxlen):
    # not fixed yet
    sbj = scipy.io.loadmat(direc_cond + '/' + lom[indexforlom])['eeg']['sbj']
    action_hist1 = sbj[0, 0][:, 3 + 1]
    action_hist2 = sbj[0, 0][:, 3 + 2]
    Goal = sbj[0,0][:, 17]

    trials = int(action_hist1.shape[0])
    labels = []
    for i in range(len(action_hist1)):
        labels.append([int(action_hist1[i]) , int(Goal[i])])
        labels.append([int(action_hist2[i]) , int(Goal[i])])
    labels = np.array(labels)

    return labels[:maxlen,:]


def ext_label_lstm(direc_cond, lom, indexforlom ,label_cond,maxlen):
    # sta memnas the state you are interested in. (1 or 2)
    # label_cond means the condition of labeling data. e.g. "maxrel" means that you want to label eeg data with maxinvfano in the sbj.regressor
    # if you want to change this label_cond make sure that you coded in the dataLoader.ext_label
    # "maxrel" : maximum reliability (wherther that maximum reliabiltiy is came from MB or MF
    # "goal"   : goal condition (specific versus flexible
    # "Pmb5"   : label trials with pmb exceed 0.5 as mb and label as mf for others.
    # "action" : label for right and left action chosen
    behav = scipy.io.loadmat(direc_cond +'/'+ lom[indexforlom] )['eeg']['sbj']

    if '/sta1' in direc_cond:
        locking = 1
    elif '/sta2' in direc_cond:
        locking = 2
    else:
        print("ERROR!!!!???? : CHECK LOCKING EVENT? E.G. STA1?")

    sbj = scipy.io.loadmat(direc_cond +'/'+ lom[indexforlom] )['eeg']['sbj']
    action_hist = sbj[0, 0][:, 5 + locking]

    trials = int(action_hist.shape[0])
    m1_indice = np.where(action_hist == 1)
    m2_indice = np.where(action_hist == 2)

    labels = np.zeros((trials,))
    labels[m1_indice] = 0
    labels[m2_indice] = 1

    return labels[:maxlen]