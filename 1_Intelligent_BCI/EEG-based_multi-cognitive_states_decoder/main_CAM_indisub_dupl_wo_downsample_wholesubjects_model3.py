import tensorflow as tf
import numpy as np
from datetime import datetime
import json
from dataLoader_model3 import *
from featureExt import *
from featureSel import *
from modelDef_biLSTM_model3 import *

import time
from os import path
import os
from shutil import copyfile

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.FATAL)

'''
# it trains and tests model.
def train_model(X_train, X_test, y_train, y_test, save_direc, subject=1, cv=1):

    # SCREENING or not : Screening means running for each of all possible features.
    if meta_json["SCREENING"] is False: # CNN, DNN etc
        # need to be fixed.. didnt cared enough at this moment [2017.12]
        print('*' * 30)
        print('*' * 30)
        print('*' * 30)
        print('**{0}'.format(meta_json["MODEL_NAME"]) + '_TRAINING')

        meta_json["rest"] = False
        print('**[{0}'.format(X_train.shape[0]) + '] samples')

        raw = 'raw'
        weighted = 'weighted'

        # W/O weight
        model1 = eval(meta_json["MODEL_NAME"] + "(meta_json, save_direc, {0}, {1}, {2})".format(subject, cv, raw))
        model1.initialize_variables()
        model1.train(X_train, y_train, X_test, y_test)
        raw_acc = model1.acc

        # WITH given weight
        model2 = eval(meta_json["MODEL_NAME"] + "(meta_json, save_direc, {0}, {1}, {2})".format(subject, cv, weighted))
        model2.initialize_variables()
        model2.saver.restore(model2.sess, meta_json["CKPT_PATH"] + meta_json["CKPT_NAME"])
        model2.train(X_train, y_train, X_test, y_test) # x : n*4*121*121, y : n
        weighted_acc = model2.acc

        return raw_acc, weighted_acc

    else :
        pass
        return 0, 0
'''


if __name__ == '__main__':
    # load initialized default setting. if you want to change model, then you have to edit the parameters_eeg.json in the root directory
    with open('parameters_eeg_dupl_wo_downsample_wholesubject_model3.json') as param_setting:
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

    acc_mat=[]

    if meta_json["SUBWISE"] is True:
        # should be fixed [2018.03.26]
        datasets = 6
        for i in range(datasets):

            print('-' * 10, ' : [', i + 1, '|', datasets, ']', '-' * 10)

            try:
                os.mkdir(save_direc + '/dataset{}'.format(i+1))
            except:
                print('yep!')

            lb = ext_label(meta_json["DATA_PATH"], ['pjs'], i+1, 'null')
            ep = ext_epochs(meta_json["DATA_PATH"], ['pjs'], i+1)

            # minibatch for test run
            # lb = lb[195:245]
            # ep = ep[:, :, 195:245]

            X = eval('ext_' + meta_json["FEATURE_TO_USE"].lower() + '(ep)')  # n_sample * 4(ch) * 121 * 121
            Y = lb  # n_sample

            # model generation for weighted gap retrieve
            model = eval(meta_json["MODEL_NAME"] + "(meta_json, save_direc)")
            model.initialize_variables()
            model.saver.restore(model.sess, meta_json["CKPT_PATH"] + meta_json["CKPT_NAME"])
            gap = model.get_gap(X, Y)
            del model

            kf = KFold(n_splits=5, shuffle=True)
            kf.get_n_splits(Y)
            accuracy_list = []
            ind_rp = 0
            for train_ind, test_ind in kf.split(Y):
                X_train, X_test = X[train_ind, :, :, :], X[test_ind, :, :, :]
                y_train, y_test = Y[train_ind], Y[test_ind]
                g_train, g_test = gap[train_ind], gap[test_ind]
                meta_json["rest"] = False
                ind_rp += 1
                print('**[{0}'.format(X_train.shape[2]) + '] samples and [{0}'.format(ind_rp) + '] CV')

                # model generation with retrieved gap
                model = eval(
                    meta_json["MODEL_NAME"] + "(meta_json, save_direc, {0}, {1}, gap={2})".format(i+1, ind_rp, True))
                model.initialize_variables()
                model.set_gap(gap)
                model.train(X_train, y_train, X_test, y_test, g_train, g_test)
                acc = model.acc

                accuracy_list.append(acc)
                filename10 = save_direc + '/dataset{}'.format(i+1) + '/acc_{}'.format(ind_rp)
                np.save(filename10, acc)

            filename1 = save_direc + '/dataset{}'.format(i+1) + '/acc'
            np.save(filename1, accuracy_list)


    else:
        print('The case of subwise==false will not be considered')

