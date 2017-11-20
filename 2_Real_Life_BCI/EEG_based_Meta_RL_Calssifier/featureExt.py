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

def ext_spectrogram_wtpfp(epoch, fs=500, window='hamming', nperseg=250, noverlap=245, nfft=500):
    # extract sepctrogram with time point / frequency point
    f, t, Sxx = signal.spectrogram(epoch, fs, window='hamming', nperseg=250, noverlap=245, nfft=500)
    e = np.where(f>=50)
    freq = f[0:e[0][0]]
    tfreq = Sxx[0:e[0][0],:]
    return t, freq, tfreq

def ext_spectrogram(epoch, fs=500, window='hamming', nperseg=250, noverlap=245, nfft=500):
    # extract sepctrogram with time point
    f, t, Sxx = signal.spectrogram(epoch, fs, window='hamming', nperseg=250, noverlap=245, nfft=500)
    e = np.where(f>=50)
    freq = f[0:e[0][0]]
    tfreq = Sxx[0:e[0][0],:]
    return tfreq

def ext_spectrogram_batch(epoch, fs=500, window='hamming', nperseg=250, noverlap=245, nfft=500):
    # extract sepctrogram with time point
    # epoch shape = (chno, timeseries, tirals)
    dat = [] # datshape = (trials , features)
    for i in range(epoch.shape[2]):
        tfreq = []
        for j in range(epoch.shape[0]):
            tfreq.append(ext_spectrogram(epoch[j,:,i], fs=500, window='hamming', nperseg=250, noverlap=245, nfft=500).flatten())
        dat.append(np.asarray(tfreq).flatten())
    return np.asarray(dat)

def ext_erp(epoch,nperseg=50, noverlap=25):
    chan_dat = []
    for j in range(epoch.shape[2]):
        maxwinlen = nperseg
        minwinlen = 0
        tfreq = []
        for i in range(epoch.shape[0]):
            while maxwinlen < epoch.shape[1]-1 and maxwinlen - minwinlen is nperseg:
                tfreq.append(epoch[i,minwinlen:maxwinlen,j])
                minwinlen += nperseg-noverlap
                maxwinlen += nperseg-noverlap
        chan_dat.append(tfreq)
    return np.array(chan_dat)


def ext_erp_mvav_batch(epoch, nperseg=50, noverlap=25):
    chan_dat=[]
    for j in range(epoch.shape[2]):
        maxwinlen = nperseg
        tfreq = []
        for i in range(epoch.shape[0]):
            while maxwinlen < epoch.shape[1]:
                maxwinlen += nperseg-noverlap
                tfreq.append(epoch[i, :maxwinlen, j].mean())
        chan_dat.append(tfreq)
    return np.array(chan_dat)
