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

    # for i in range(10):
        with open(filelist[i], 'rb') as f: plk_tmp = pkl.load(f)
        ecg_re = ecg.ecg(signal=plk_tmp, sampling_rate=Fs, show=False)
        rpeaks_tmp = ecg_re['rpeaks'].tolist()
        nni = tools.nn_intervals(rpeaks=rpeaks_tmp)
        nni_tmp = nni.reshape((-1, int(nni.shape[0])))  # for 2d data type
        rp = RecurrencePlot(threshold='point', percentage=20)
        X_rp = rp.fit_transform(nni_tmp)
        dst = cv2.resize(X_rp[0], dsize=(135, 135), interpolation=cv2.INTER_AREA)
        shape_tmp.append(X_rp.shape)
        recurrence_tmp.append(X_rp)
        recur_resize.append(dst)
        # for pandas
        # shape_tmp = shape_tmp.append(pd.DataFrame(X_rp.shape))
        # plot check
        plt.imshow(X_rp[0], cmap='binary', origin='lower')
        plt.plot(nni)
        plt.title('Recurrence Plot', fontsize=16)
        plt.tight_layout()
        plt.show()
        # np_tmp = np.column_stack([np_tmp, X_rp])
        if i == 0:
            pass
    return shape_tmp, recurrence_tmp, np.asarray(recur_resize)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, Convolution2D, MaxPooling2D

model = Sequential()

model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(1, 32, 32), data_format='channels_first'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Convolution2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
# model.add(LeakyReLU(alpha=0.03))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

#reshape to include depth
X_train = x_train.reshape(x_train.shape[0], 1, 32,32)
#convert to float32 and normalize to [0,1]
X_train = X_train.astype('float32')
X_train /= np.amax(X_train)
# convert labels to class matrix, one-hot-encoding
Y_train = np_utils.to_categorical(y_train, 3)
# split in train and test set
X_train, x_test, Y_train, y_test = train_test_split(X_train, Y_train, test_size=0.1)

model.fit(X_train, Y_train, epochs=200, batch_size=16,shuffle=True)
