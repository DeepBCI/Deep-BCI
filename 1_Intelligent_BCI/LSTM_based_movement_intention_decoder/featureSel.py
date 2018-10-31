from sklearn import svm
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

def channel_selection(data,chan_arr):
    return data[chan_arr,:,:]


def mutual_info(data, label, k=10000):
    return SelectKBest(MINF, k).fit_transform(data, label)

def fisher_ratio(data,label,k=10000):
    m1 = np.where(label == 1)[0] # label == 1
    m2 = np.where(label == -1)[0] # label == -1
    fr = np.zeros((1,data.shape[1]))
    for i in range(data.shape[1]):
        fr[0,i] = ( np.mean(data[m1,i]) - np.mean(data[m2,i]) ) ** 2   / (np.std(data[m1,1])**2 + np.std(data[m2,1])**2 )
    sorted_fr = np.sort(fr)
    feat_index = np.zeros(k)
    for j in range(k):
        feat_index[j] = np.where(fr[0,:]==sorted_fr[0,-1*(1+j)])[0][0]
    out = []
    for k in range(data.shape[0]):
        out.append(data[k,feat_index.astype(int).tolist()])
    return np.asarray(out)

def chi2(data, label, k=10000):
    return SelectKBest(chi2, k).fit_transform(data, label)