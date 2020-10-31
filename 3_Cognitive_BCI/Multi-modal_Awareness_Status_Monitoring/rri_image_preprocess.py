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



# 자료구조?
if __name__ == "__main__":
    result, recur_result, awake_recur_ressize_result = rri_test_recurrent(dir_list1)   # awake
    print(len(result))
    # print(result[0])
    pd_result =  pd.DataFrame(result)
    # print("result=", pd_result)
    """
    # plot check
    plot_tmp = recur_result[50]
    plot_tmp = plot_tmp[0]
    print(plot_tmp.shape)
    # matrix 채울 방법이...
    dst = cv2.resize(plot_tmp, dsize=(135, 135), interpolation=cv2.INTER_AREA)
    print(type(dst), dst.shape)
    """
    """
    cv2.imshow('dst', dst)
    # cv2.imshow('dst', dst)
    plt.imshow(plot_tmp, cmap='binary', origin='lower')
    plt.title('Recurrence Plot', fontsize=16)
    plt.tight_layout()
    plt.show()
    #
    # cv2.waitKey(0)
    """
    print(awake_recur_ressize_result.shape)
    print(awake_recur_ressize_result[0])
    # Create List of Single Item Repeated n Times in Python
    # ref: https://stackoverflow.com/questions/3459098/create-list-of-single-item-repeated-n-times
    awake_length = len(dir_list1)
    print("awake_length=", type(awake_length), awake_length)
    y_awake = ["awake"] * awake_length
    print(len(y_awake), y_awake[0])

    _, _, drowsy_recur_ressize_result = rri_test_recurrent(dir_list2)  # drowsy
    drowsy_length = len(dir_list2)
    print("drowsy_length=", type(drowsy_length), drowsy_length)
    y_drowsy = ["drowsy"] * drowsy_length
    print(len(y_drowsy), y_drowsy[0])

    _, _, uncons_recur_ressize_result = rri_test_recurrent(dir_list3)  # uncons
    uncons_length = len(dir_list3)
    print("drowsy_length=", type(uncons_length), uncons_length)
    y_uncons = ["uncons"] * uncons_length
    print(len(y_uncons), y_uncons[0])

    print("final numpy x shape=", np.asarray(recur_resize).shape)

    Y = y_awake + y_drowsy + y_uncons
    print("finaly list y len=", len(Y))
    Y = np.asarray(Y)
    print(Y.shape)

    # make file! in order to saive time
    X = np.asarray(recur_resize)
    np.save('recur_X_np', X)
    np.save('recur_Y_np', Y)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.30)

    """
    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1. / 255)


    train_datagen.flow_from_directory()
    """
    # train_datagen.flow_from_dataframe()
    cv2.destroyAllWindows()