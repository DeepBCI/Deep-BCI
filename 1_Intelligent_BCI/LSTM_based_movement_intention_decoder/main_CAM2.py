import tensorflow as tf
import numpy as np
from datetime import datetime
import json
from dataLoader import *
from featureExt import *
from featureSel import *
from modelDef import *
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
def train_model(X_train, X_test, y_train, y_test, save_direc):

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
    X_train = eval ( 'ext_' + meta_json["FEATURE_TO_USE"].lower() + '(X_train)' )
    X_test = eval ( 'ext_' + meta_json["FEATURE_TO_USE"].lower() + '(X_test)' )
    # mbraw2 = eval ( 'ext_' + meta_json["FEATURE_TO_USE"].lower() + '(m1_sta2dat)' )
    # mfraw2 = eval ( 'ext_' + meta_json["FEATURE_TO_USE"].lower() + '(m2_sta2dat)' )

    # mblbl = np.ones((mbraw.shape[0]))
    # mflbl = np.ones((mfraw.shape[0])) * (-1)
    #
    # mblbl2 = np.ones((mbraw2.shape[0]))
    # mflbl2 = np.ones((mfraw2.shape[0])) * (-1)

    # full_data = np.concatenate((mbraw, mfraw), 0)
    # full_label = np.concatenate((mblbl, mflbl), 0)

    # full_data2 = np.concatenate((mbraw2, mfraw2), 0)
    # full_label2 = np.concatenate((mblbl2, mflbl2), 0)

    # SCREENING or not : Screening means running for each of all possible features.
    if meta_json["SCREENING"] is False: # CNN, DNN etc
        # need to be fixed.. didnt cared enough at this moment [2017.12]
        print('*' * 30)
        print('*' * 30)
        print('*' * 30)
        print('**{0}'.format(meta_json["MODEL_NAME"]) + '_TRAINING')
        accuracy_tw = []

        accuracy_list = []
        epoch_i = 0

        meta_json["rest"] = False
        print('**[{0}'.format(X_train.shape[0]) + '] samples')

        model1 = eval(meta_json["MODEL_NAME"] + "(meta_json)")
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

    else :
        accuracy_tw = []
        print('*' * 30)
        print('*' * 30)
        print('*' * 30)
        print('**{0}Screening ? NO!!!')
        # for cc in range(m1_sta1dat.shape[0]):
        #     print('**CH' + '[ {0}/{1} ]'.format(cc+1, m1_sta1dat.shape[0]))
        #     for ii in range(mbraw.shape[2]): # for each feature vector,
        #         print(' '*4 + '[ {0}'.format(ii + 1) + '/' + '{0} ]'.format(mbraw.shape[2]))
        #         data = full_data[:, cc, ii, :]
        #         data_folded = KFold(n_splits=10, shuffle=True)
        #         data_folded.get_n_splits(data)
        #         accuracy_list = []
        #         start_time = time.time()
        #         for train_index, test_index in data_folded.split(data):
        #             X_train, X_test = data[train_index], data[test_index]
        #             y_train, y_test = full_label[train_index], full_label[test_index]
        #             model1 = eval(meta_json["MODEL_NAME"] + "(meta_json)")
        #             model1.train(X_train, y_train)
        #             accuracy_list.append(model1.accuracy_score(X_test, y_test))
        #
        #         data = full_data2[:, cc, ii, :]
        #         data_folded = KFold(n_splits=10, shuffle=True)
        #         data_folded.get_n_splits(data)
        #         accuracy_list = []
        #         start_time = time.time()
        #         for train_index, test_index in data_folded.split(data):
        #             X_train, X_test = data[train_index], data[test_index]
        #             y_train, y_test = full_label2[train_index], full_label2[test_index]
        #             model2 = eval(meta_json["MODEL_NAME"] + "()")
        #             model2.train(X_train, y_train)
        #             accuracy_list.append(model2.accuracy_score(X_test, y_test))
        #         accuracy_tw.append(accuracy_list)
        return accuracy_tw



if __name__ == '__main__':
    # load initialized default setting. if you want to change model, then you have to edit the parameters_eeg.json in the root directory
    with open('parameters_eeg.json') as param_setting:
        meta_json = json.load(param_setting)
    save_direc = meta_json["MODEL_NAME"] + '-' + meta_json["STARTED_DATESTRING"]

    # restoring or saving
    if path.isfile(save_direc + "/model.ckpt.meta") is True :
        meta_json["rest"] = True
    else:
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
            ep = ext_epochs(meta_json["DATA_DIRECTORY"] + BN + '/sta1', ed_list1, i)
            ep2 = ext_epochs(meta_json["DATA_DIRECTORY"] + BN + '/sta2', ed_list2, i)
            # cutting by minimum size
            # min1 = min([mbep.shape[2], mfep.shape[2]])
            # min2 = min([mbep2.shape[2], mfep2.shape[2]])
            # mbep = mbep[:, :, :min1]
            # mfep = mfep[:, :, :min1]
            # mbep2 = mbep2[:, :, :min2]
            # mfep2 = mfep2[:, :, :min2]


            # Labels of the data
            # label_cond means the condition of labeling data. e.g. "maxrel" means that you want to label eeg data with maxinvfano in the sbj.regressor
            # if you want to change this label_cond make sure that you coded in the dataLoader.ext_label
            # "maxrel" : maximum reliability (wherther that maximum reliabiltiy is came from MB or MF
            # "goal"   : goal condition (specific versus flexible
            # "Pmb5"   : label trials with pmb exceed 0.5 as mb and label as mf for others.
            # "action" : label for right and left action chosen
            lb  = ext_label(meta_json["DATA_DIRECTORY"] + BN + '/sta1',ed_list1,i, meta_json["LABEL"])
            lb2 = ext_label(meta_json["DATA_DIRECTORY"] + BN + '/sta2',ed_list2,i, meta_json["LABEL"])

            # model training
            # [m1_sta1dat, m2_sta1dat, m1_sta2dat, m2_sta2dat] = [mbep, mfep, mbep2, mfep2]
            meta_json["rest"] = False

            accuracy_tw = train_model(ep, ep2, lb, lb2)
            filename1 = save_direc + '/acc_sub' + str(i+1)
            np.save(filename1, accuracy_tw)
    else:
        ep = ext_epochs(meta_json["DATA_DIRECTORY"] + BN + '/sta1', ed_list1, 0)
        ep2 = ext_epochs(meta_json["DATA_DIRECTORY"] + BN + '/sta2', ed_list2, 0)
        lb = ext_label(meta_json["DATA_DIRECTORY"] + BN + '/sta1', ed_list1, 0, meta_json["LABEL"])
        lb2 = ext_label(meta_json["DATA_DIRECTORY"] + BN + '/sta2', ed_list2, 0, meta_json["LABEL"])

        print('-' * 10, ' : [',  1, '|', np.array(ed_list1).shape[0] , ']', '-' * 10)
        # Feature load from whole subject data
        for i in range(np.asarray(ed_list1).shape[0] - 1):
            print('-' * 10, ' : [', i + 2, '|', np.array(ed_list1).shape[0], ']', '-' * 10)
            ep = np.concatenate((ep, ext_epochs(meta_json["DATA_DIRECTORY"] + BN + '/sta1', ed_list1, i + 1)), 2)
            ep2 = np.concatenate((ep2, ext_epochs(meta_json["DATA_DIRECTORY"] + BN + '/sta2', ed_list2, i + 1)), 2)
            lb = np.concatenate((lb, ext_label(meta_json["DATA_DIRECTORY"] + BN + '/sta1', ed_list1, i+1, meta_json["LABEL"])),0)
            lb2 = np.concatenate((lb2, ext_label(meta_json["DATA_DIRECTORY"] + BN + '/sta2', ed_list2, i + 1, meta_json["LABEL"])), 0)

        # globalized meta_json (parameter setting)
        # global meta_json

        # cutting by minimum size
        min = min( [ep.shape[2], ep2.shape[2]] )
        ep = ep[:,:,:min]
        ep2 = ep2[:,:,:min]
        lb = lb[:min]
        lb2 = lb2[:min]


#        ep = ep[:,:,:100]
#        ep2 = ep2[:,:,:100]
#        lb = lb[:100]
#        lb2 = lb2[:100]

        ep = np.concatenate ( (ep, ep2), axis=2)
        lb = np.concatenate ( (lb, lb2), axis=0)

        # perm_ind = np.random.choice(range(lb.shape[0]), lb.shape[0], replace=False)
        # ep = ep[:,:,perm_ind]
        # lb = lb[perm_ind]

        kf = KFold(n_splits=10, shuffle=True)
        kf.get_n_splits(lb)
        ind_rp = 0
        accuracy_list =[]
        for train_ind, test_ind in kf.split(lb):
            X_train, X_test = ep[:, :, train_ind], ep[:, :, test_ind]
            y_train, y_test = lb[train_ind], lb[test_ind]
            meta_json["rest"] = False
            ind_rp += 1
            print('**[{0}'.format(X_train.shape[2]) + '] samples and [{0}'.format(ind_rp) + '] CV')
            accuracy_tw = train_model(X_train, X_test, y_train, y_test, save_direc)
            accuracy_list.append(accuracy_tw)

            filename10 = save_direc + '/acc_{0}'.format(ind_rp)
            np.save(filename10, accuracy_tw)
        # model training
        # [m1_sta1dat, m2_sta1dat, full_label_m1, full_label_m2] = [ep, ep2, lb, lb2]
        filename1 = save_direc + '/acc'
        np.save(filename1,accuracy_list)





