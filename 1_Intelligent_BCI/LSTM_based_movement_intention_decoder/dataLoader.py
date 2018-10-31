
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
    lom.sort()
    epochs = scipy.io.loadmat(direc_cond +'/'+ lom[indexforlom] )['eeg']['epoch'][0,0]
    return epochs

def ext_label(direc_cond, lom, indexforlom ,label_cond):
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

    if label_cond.lower() == 'maxrel':
        reg = scipy.io.loadmat(direc_cond +'/'+ lom[indexforlom] )['eeg']['reg']
        m1 = reg[0, 0][0, 24][0, 0][1][6,:]
        m2 = reg[0, 0][0, 25][0, 0][1][6,:]
        mm = reg[0, 0][0, 21][0, 0][1][6,:]

        trials = int(m1.shape[0]/3)
        indices = np.int16(np.linspace(0,trials-1,trials)*3 + locking)
        m1_real = m1[indices]
        m2_real = m2[indices]
        mm_real = mm[indices]

        m1_indice = np.where(mm_real == m1_real)[0]
        m2_indice = np.where(mm_real == m2_real)[0]
    # elif label_cond.lower() == 'goal':
    # elif label_cond.lower() == 'pmb5':
    elif label_cond.lower() == 'action':
        sbj = scipy.io.loadmat(direc_cond +'/'+ lom[indexforlom] )['eeg']['sbj']
        action_hist = sbj[0, 0][:, 5 + locking]

        trials = int(action_hist.shape[0])
        m1_indice = np.where(action_hist == 1)
        m2_indice = np.where(action_hist == 2)


    else :
        # whatelse?
        print("ERROR!!!!???? : CHECK YOUR LABEL FROM '''PARAMETERS_EEG.JSON''' FILE")

    labels = np.zeros((trials,))
    labels[m1_indice] = 0
    labels[m2_indice] = 1
    return labels



def ext_input_lstm(direc_cond, lom, indexforlom, maxlen):
    behav = scipy.io.loadmat(direc_cond +'/'+ lom[indexforlom] )['eeg']['sbj']

    if '/sta1' in direc_cond:
        locking = 1
    elif '/sta2' in direc_cond:
        locking = 2
    else:
        print("ERROR!!!!???? : CHECK LOCKING EVENT? E.G. STA1?")

    sbj = scipy.io.loadmat(direc_cond +'/'+ lom[indexforlom] )['eeg']['sbj']
    Sta = sbj[0, 0][:, 3 + locking]
    Goal = sbj[0,0][:, 17]
    labels = np.concatenate((Sta.reshape((-1, 1)), Goal.reshape((-1, 1))), axis=1)

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