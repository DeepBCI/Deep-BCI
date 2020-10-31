import numpy as np
import pandas as pd
import pickle as pkl
from util.file_manager import FileManager
import pyhrv.tools as tools
from biosppy.signals import ecg
import sklearn as sk
from pyts.image import RecurrencePlot
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import sklearn
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
from keras.preprocessing.image import ImageDataGenerator



pd.set_option('display.max_colwidth'    , 40) ## 각 컬럼 width 최대로
pd.set_option('display.max_rows', 500) ## rows 500
pd.set_option('display.max_columns', 500) ## columns
pd.set_option('display.width', 1000)

Fs = 250

filepath_awake = 'd:/bci/ecg_all/pkl-250Hz/without bed/awake'
filepath_drows = 'd:/bci/ecg_all/pkl-250Hz/without bed/drows'
filepath_uncons = 'd:/bci/ecg_all/pkl-250Hz/without bed/uncons'

dir_list1 = list()
dir_list2 = list()
dir_list3 = list()

fm = FileManager()
print("file manager start")
fm.search(filepath_awake, dir_list1)
dir_list1 = dir_list1[:10]
fm.search(filepath_drows, dir_list2)
dir_list2 = dir_list2[:10]
fm.search(filepath_uncons, dir_list3)
dir_list3 = dir_list3[:10]
print('data length=', len(dir_list1), len(dir_list2), len(dir_list3))
# np_tmp = np.empty(shape=(10,140,140))

shape_tmp = list()
recurrence_tmp = list()
recur_resize = list()

def rri_test_recurrent(filelist=None):
    global shape_tmp
    for i in range(len(filelist)):
    # global  np_tmp

    # for i in range(10):
        with open(filelist[i], 'rb') as f: plk_tmp = pkl.load(f)
        ecg_re = ecg.ecg(signal=plk_tmp, sampling_rate=Fs, show=False)
        # print(ecg_re)
        rpeaks_tmp = ecg_re['rpeaks'].tolist()
        nni = tools.nn_intervals(rpeaks=rpeaks_tmp)
        # print(nni.shape, type(nni))
        # print(nni)
        nni_tmp = nni.reshape((-1, int(nni.shape[0])))  # for 2d data type
        # rp = RecurrencePlot(threshold='distance', percentage=20)
        rp = RecurrencePlot(threshold='point', percentage=20)
        X_rp = rp.fit_transform(nni_tmp)
        # print("X_rp shape=", len(filelist)-i)
        # print(type(X_rp), X_rp.shape)
        # print(type(X_rp[0]), X_rp[0].shape)
        # for list
        # one of the most important parts
        dst = cv2.resize(X_rp[0], dsize=(135, 135), interpolation=cv2.INTER_AREA)