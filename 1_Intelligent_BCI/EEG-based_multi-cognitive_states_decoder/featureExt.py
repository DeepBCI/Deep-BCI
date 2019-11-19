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
    # need to be fixed [2017.12]
    f, t, Sxx = signal.spectrogram(epoch, fs, window='hamming', nperseg=250, noverlap=245, nfft=500)
    e = np.where(f>=50)
    freq = f[0:e[0][0]]
    tfreq = Sxx[0:e[0][0],:]
    return t, freq, tfreq

def ext_spectrogram(epoch, fs=1000, window='hamming', nperseg=2000, noverlap=1975, nfft=3000):
    #used to have default of fs=500, nperseg=250, nopverlap=225, nfft=500

    # epoch.shape = channel number, timepoint, trials
    # extract sepctrogram with time point
    # nperseg : length of each segment.
    # noverlap : number of points to overlap between segments.
    # nfft : length of FFT, if it is different from nperseg, that means zero padded manner.
    # default means ft for 500ms window data with 50ms overlapping sliding window.
    # as a result of default setting, t.shape = (61,)
    # the frequency domain cut by the shape of time domain

    dat = [] # datshape = (trials , features)
    for i in range(epoch.shape[2]):
        tfreq = []
        for j in range(epoch.shape[0]):
            f, t, Sxx = signal.stft(epoch[j,:,i], fs=fs, window=window, nperseg=nperseg, noverlap=noverlap, nfft=nfft)
            interval = f[-1]/(len(f)-1)
            req_len = int(40 / interval)
            # tfreq.append(np.abs(Sxx[:t.shape[0], :]).transpose())
            tfreq.append(np.abs(Sxx[:121, :]).transpose())

        dat.append(np.asarray(tfreq))

    return np.array(dat) # shape : (trials, channel number, time, freq), and by default, time and freq : 121, 121

def ext_spectrogram_original(epoch, fs=500, window='hamming', nperseg=250, noverlap=225, nfft=500):
    # epoch.shape = channel number, timepoint, trials
    # extract sepctrogram with time point
    # nperseg : length of each segment.
    # noverlap : number of points to overlap between segments.
    # nfft : length of FFT, if it is different from nperseg, that means zero padded manner.
    # default means ft for 500ms window data with 50ms overlapping sliding window.
    # as a result of default setting, t.shape = (61,)
    # the frequency domain cut by the shape of time domain

    dat = [] # datshape = (trials , features)
    for i in range(epoch.shape[2]):
        tfreq = []
        for j in range(epoch.shape[0]):
            f, t, Sxx = signal.stft(epoch[j,:,i], fs=fs, window=window, nperseg=nperseg, noverlap=noverlap, nfft=nfft)

            # tfreq.append(np.abs(Sxx[:t.shape[0], :]).transpose())
            tfreq.append(np.abs(Sxx[:61, :]).transpose().repeat(2, axis=1)[:, :t.shape[0]])

        dat.append(np.asarray(tfreq))

    return np.array(dat) # shpae of (trials, channel number, time, freq)

def ext_erp(epoch,nperseg=50, noverlap=25): # epoch size  (channel) * (time point) * (trials)
    chan_dat = []
    for j in range(epoch.shape[2]): # trials
        tfreq = []
        for i in range(epoch.shape[0]): # channel
            chan = []
            maxwinlen = nperseg
            minwinlen = 0
            while maxwinlen < epoch.shape[1]-1 and maxwinlen - minwinlen is nperseg:
                chan.append(epoch[i,minwinlen:maxwinlen,j])
                minwinlen += nperseg-noverlap
                maxwinlen += nperseg-noverlap

            tfreq.append(np.array(chan))
        chan_dat.append(tfreq)
    return np.array(chan_dat) # shape of (trials, channel, # of time window (58 fir 100ms window with 50ms overlapping), 50 (length of time window)



def ext_spectrogram_batch(epoch, fs=500, window='hamming', nperseg=250, noverlap=245, nfft=500):
    # extract sepctrogram with time point
    # epoch shape = (chno, timeseries, tirals)
    # need to be fixed [2017.12]

    dat = [] # datshape = (trials , features)
    for i in range(epoch.shape[2]):
        tfreq = []
        for j in range(epoch.shape[0]):
            tfreq.append(ext_spectrogram(epoch[j,:,i], fs=500, window='hamming', nperseg=250, noverlap=245, nfft=500).flatten())
        dat.append(np.asarray(tfreq).flatten())
    return np.asarray(dat)

def ext_erp_mvav_batch(epoch, nperseg=50, noverlap=25):
    # need to be fixed [2017.12]

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


######################
# Newly Defined
# 19.01.12
# PJS
######################

def ext_upsampling(ndarr, n):
    # ndarr : training data of 2D ndarray type
    # n : multiple factor (must be a positive integer)

    err_flag = False

    if not ndarr.shape == (2,2):
        print('DimensionError at ext_upsampling 1st parameter :', ndarr.shape, '(should be 2D)')
        err_flag = True

    if not float(n).is_integer() or not n > 0:
        print('TypeError at ext_upsampling 2nd parameter :',n,'(should be a positive integer)')
        err_flag = True

    if err_flag:
        exit(-1)

    return np.kron(ndarr, np.ones((n,n)))

def ext_downsampling(ndarr, n):
    # ndarr : training data of 2D ndarray type
    # n : divide factor (must be a positive integer and must be a divisor of the number of row/col)

    err_flag = False
    chk_2nd = True

    if not ndarr.shape == (2, 2):
        print('DimensionError at ext_downsampling 1st parameter :', ndarr.shape, '(should be 2D)')
        err_flag = True
        chk_2nd = False
    elif not ndarr.shape[0] != ndarr.shape[1]:
        print('SizeError at ext_downsampling 1st parameter : [', ndarr.shape[0],',',ndarr.shape[1], '] (number of row and column should be the same)')
        err_flag = True
        chk_2nd = False

    if not float(n).is_integer() or not n > 0:
        print('TypeError at ext_downsampling 2nd parameter :', n, '(should be a positive integer)')
        err_flag = True
    elif chk_2nd and not ndarr.shape[0] % n == 0:
        print('DivideError at ext_downsampling 2nd parameter :', n, '(should be a positive integer)')
        err_flag = True

    if err_flag:
        exit(-1)

    (row, col) = ndarr.shape
    result = np.zeros((row/n, col/n))

    for r in (row/n):
        for c in range(col/n):

            frag = [0]*n
            for i in range(n):
                for j in range(n):
                    frag[i*n+j] = ndarr[r*n+i,c*n+j]

            result[r,c] = np.mean(frag)

    return result

def ext_fftfreq(epoch):
    channel = np.array(
        [63, 62, 64, 17, 10, 5, 1, 61, 8, 11, 2, 18, 13, 9, 6, 3, 59, 58, 12, 60, 19, 14, 57, 56, 23, 15, 7, 54, 53,
         55])
    channel[:] = channel[:] - 1

    dat = []
    for i in range(epoch.shape[2]):
        ftfreq = []
        for ch in channel:
            power = np.abs(np.fft.fft(epoch[ch, :, i]))
            ftfreq.append(power[:120].transpose())

        dat.append(np.asarray(ftfreq))

    return np.array(dat)


