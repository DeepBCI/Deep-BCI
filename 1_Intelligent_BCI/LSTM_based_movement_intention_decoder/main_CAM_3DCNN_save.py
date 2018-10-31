import tensorflow as tf
import numpy as np
from datetime import datetime
import json
from dataLoader import *
from featureExt import *
from featureSel import *
from modelDef_LSTM import *
import time
from os import path
import os
from shutil import copyfile

BATCH_SIZE = 1
DATA_DIRECTORY = '../eeg_save/'
LOGDIR_ROOT = 'log'
CHECK_OR_NOT = False
CHECKPOINT_EVERY = 1000
NUM_STEPS = int(1e4)  # int(1e5)
LEARNING_RATE = 1e-3
STARTED_DATESTRING = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
SAMPLE_SIZE = 1000
EPSILON = 0.001
MOMENTUM = 0.9
NUM_OF_CHAN = 64
FEATURE_TO_USE = 'ERP' # ERP or spectrogram specto
MODEL_NAME = 'SVM' # CNN, SVM, CNN_CAM
SCREENING = True # Screening means you dont want to apply any specified feature selection algorithm.
# if you dont want to use screening, you have to specify the usage of channel (CHAN_WISE)
# that means, if Screening is true, then you have to



# it trains and tests model.
def train_model( X_train, X_test, y_train, y_test, save_direc, model1):

    # initialize
    mbraw=[]
    mfraw=[]
    mbraw2=[]
    mfraw2=[]

    # feature extraction : shape = {trials, concatenated shape, # of feature vectors, # of features}
    #       # of feature vectos are only required if and only if you are screening all of possible features.
    #       if you want to apply any specific feature selection, please faltten this part.
    # Default ERP setting is 100ms (nperseg=50, noverlap=25) if you want to change any of this, please change in the featureExt file
    # if, input arguement of ext_ is multi channel data, then you will receive output as shape of (trials, channel*# of feature vector, #of features)
    # shape of (#trials, #channels, #featuresVector, #featuresInVector)
    # [FEATURE_TO_USE : "ERP"]         - (#trials, #channels, #featuresVector, #featuresInVector)
    # [FEATURE_TO_USE : "spectrogram"] - (#trials, #channels, #features(time), #features(freq))
    # spectro_train = eval ( 'ext_' + meta_json["FEATURE_TO_USE"].lower() + '(spectro_train)' )
    # spectro_test = eval ( 'ext_' + meta_json["FEATURE_TO_USE"].lower() + '(spectro_test)' )

    print('*' * 30)
    print('*' * 30)
    print('*' * 30)
    print('**{0}'.format(meta_json["MODEL_NAME"]) + '_TRAINING')
    accuracy_tw = []

    accuracy_list = []
    epoch_i = 0

    meta_json["rest"] = False
    print('**[{0}'.format(X_train.shape[0]) + '] samples')

    model1.init_LSTM()
    model1.train(X_train, y_train, X_test, y_test)
    accuracy_tw = model1.accuracy_score(X_test, y_test)

    # for train_index, test_index in data_folded.split(data):
    #     epoch_i += 1
    #     print('**[{0}'.format(epoch_i) + '/10] th epoch training....')
    #     X_train, X_test = data[train_index], data[test_index]
    #     y_train, y_test = full_label[train_index], full_label[test_index]
    #     model1 = eval(meta_json["MODEL_NAME"] + "(meta_json)")
    #     model1.train(X_train, y_train)
    #     # if you don't want to restore - please make sure your meta_json["rest"] = False
    #     meta_json["rest"] = False
    #     accuracy_list.append(model1.accuracy_score(X_test, y_test))

    # # test
    # X_train = np.concatenate( (np.ones((1000,64,61,61)),  np.ones((1000,64,61,61)) *(-2)), axis=0)
    # x_train = np.concatenate( (np.ones((1000,64,61,61)),  np.ones((1000,64,61,61)) *(-2)), axis=0)
    # y_train = np.concatenate( (np.ones(1000), np.ones(1000)*(-1)) ,axis = 0)
    # X_train = data[:900,:,:,:]
    # y_train = full_label[:900]
    return accuracy_tw



if __name__ == '__main__':
    # load initialized default setting. if you want to change model, then you have to edit the parameters_eeg.json in the root directory
    with open('parameters_eeg.json') as param_setting:
        meta_json = json.load(param_setting)
    save_direc = meta_json["MODEL_NAME"] + '-' + meta_json["STARTED_DATESTRING"]

    # restoring or saving
    # if path.isfile(save_direc + "/model.ckpt.meta") is True :
    #     meta_json["rest"] = True
    # else:
    meta_json["rest"] = False


    try:
        os.mkdir(save_direc)
    except:
        print('yep!')

    # this model utilizes alpha band in case of ERP and utilzes BB(broad band) in case of spectrogram.
    if meta_json["FEATURE_TO_USE"].lower() == "erp":
        BN = 'ALPHA'
    else:
        BN = 'BB'

    copyfile('parameters_eeg.json', save_direc + '/parameters_eeg.json')

    # with open(save_direc + '/parameters_eeg.json', 'w', encoding="utf-8") as make_file:
    #     json.dump(meta_json, make_file, ensure_ascii=False, indent="\t")

    # Load file name lists
    ed_list1 = ext_files(meta_json["DATA_DIRECTORY"] + BN + '/sta1')
    ed_list2 = ext_files(meta_json["DATA_DIRECTORY"] + BN + '/sta2')

    acc_mat=[]


    if meta_json["SUBWISE"] is True:
        # should be fixed [2018.03.26]
        for i in range(np.asarray(ed_list1).shape[0]):
            print('-' * 10, ' : [', i + 1, '|', np.array(ed_list1).shape[0], ']', '-' * 10)

    else:
        ind_rp = 0

        # Feature load from whole subject data
        for i in range(np.asarray(ed_list1).shape[0] - 1):
            ep = ext_epochs(meta_json["DATA_DIRECTORY"] + BN + '/sta1', ed_list1, 0)
            ep2 = ext_epochs(meta_json["DATA_DIRECTORY"] + BN + '/sta2', ed_list2, 0)

            # maxlength
            maxtr = ep.shape[2]
            maxtr2 = ep2.shape[2]
            lb = ext_label_lstm(meta_json["DATA_DIRECTORY"] + BN + '/sta1', ed_list1, 0, meta_json["LABEL"], maxtr)
            lb2 = ext_label_lstm(meta_json["DATA_DIRECTORY"] + BN + '/sta2', ed_list2, 0, meta_json["LABEL"], maxtr2)
            inp = ext_input_lstm(meta_json["DATA_DIRECTORY"] + BN + '/sta1', ed_list1, 0, maxtr)
            inp2 = ext_input_lstm(meta_json["DATA_DIRECTORY"] + BN + '/sta2', ed_list1, 0, maxtr2)

            print('-' * 10, ' : [', 1, '|', np.array(ed_list1).shape[0], ']', '-' * 10)
            # Feature load from whole subject data
            for i in range(np.asarray(ed_list1).shape[0] - 1):
                print('-' * 10, ' : [', i + 2, '|', np.array(ed_list1).shape[0], ']', '-' * 10)
                eptemp = ext_epochs(meta_json["DATA_DIRECTORY"] + BN + '/sta1', ed_list1, i + 1)
                ep = np.concatenate((ep, eptemp), 2)
                eptemp2 = ext_epochs(meta_json["DATA_DIRECTORY"] + BN + '/sta2', ed_list2, i + 1)
                ep2 = np.concatenate((ep2, eptemp2), 2)
                maxtr = eptemp.shape[2]
                maxtr2 = eptemp2.shape[2]

                lbtemp = ext_label_lstm(meta_json["DATA_DIRECTORY"] + BN + '/sta1', ed_list1, i + 1, meta_json["LABEL"],
                                        maxtr)
                lb = np.concatenate((lb, lbtemp), 0)
                lbtemp2 = ext_label_lstm(meta_json["DATA_DIRECTORY"] + BN + '/sta2', ed_list2, i + 1,
                                         meta_json["LABEL"], maxtr2)
                lb2 = np.concatenate((lb2, lbtemp2), 0)

                inptemp = ext_input_lstm(meta_json["DATA_DIRECTORY"] + BN + '/sta1', ed_list1, i + 1, maxtr)
                inp = np.concatenate((inp, inptemp), 0)
                inptemp2 = ext_input_lstm(meta_json["DATA_DIRECTORY"] + BN + '/sta2', ed_list2, i + 1, maxtr2)
                inp2 = np.concatenate((inp2, inptemp2), 0)

            # globalized meta_json (parameter setting)
            # global meta_json

            # cutting by minimum size
            min = min([ep.shape[2], ep2.shape[2]])

            # min = 100
            ep = ep[:, :, :min]
            ep2 = ep2[:, :, :min]
            lb = lb[:min]
            lb2 = lb2[:min]
            inp = inp[:min]
            inp2 = inp2[:min]

            ep = np.concatenate ( (ep, ep2), axis=2)
            lb = np.concatenate ( (lb, lb2), axis=0)
            inp = np.concatenate ( (inp, inp2), axis=0)

            # perm_ind = np.random.choice(range(lb.shape[0]), lb.shape[0], replace=False)
            # ep = ep[:,:,perm_ind]
            # lb = lb[perm_ind]

            kf = KFold(n_splits=10, shuffle=False)
            kf.get_n_splits(lb)
            accuracy_list =[]
            model1 = addon_LSTM(meta_json)

            for train_ind, test_ind in kf.split(lb):
                spectro_train, spectro_test = ep[:, :, train_ind], ep[:, :, test_ind]
                spectro_train = eval('ext_' + meta_json["FEATURE_TO_USE"].lower() + '(spectro_train)')
                spectro_test = eval('ext_' + meta_json["FEATURE_TO_USE"].lower() + '(spectro_test)')
                (xtrain, xtest) = model1.puts_out(spectro_train, spectro_test)

                X_train, X_test = inp[train_ind,:], inp[test_ind,:]
                y_train, y_test = lb[train_ind], lb[test_ind]

                X_train = np.concatenate((X_train, xtrain.reshape((-1, 1))), axis=1)
                X_test = np.concatenate((X_test, xtest.reshape((-1, 1))), axis=1)

                ind_rp += 1
                print('**[{0}'.format(ind_rp) + ']')

                if ind_rp == 1:
                    x_tr_tot = X_train
                    x_te_tot = X_test
                else:
                    x_tr_tot = np.concatenate((x_tr_tot,X_train), axis=0)
                    x_te_tot = np.concatenate((x_te_tot, X_test), axis=0)

        np.save('ToT_label3D', (x_tr_tot, x_te_tot))






