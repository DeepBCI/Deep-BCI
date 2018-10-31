from featureExt import *
from dataLoader import *
from modelDef import *
from featureSel import *
import numpy as np
import os

chan_vec = [5, 10, 12, 18, 58, 60]
save_direc = 'chan_by_chan_ebe_sta12_MAXREL_wholesub'

if __name__ == '__main__':
    MBlist = ext_files('ALPHA/sta1', 'MBMAX')
    MFlist = ext_files('ALPHA/sta1', 'MFMAX')
    MBlist2 = ext_files('ALPHA/sta2', 'MBMAX')
    MFlist2 = ext_files('ALPHA/sta2', 'MFMAX')
    acc_mat=[]

    try:
        os.mkdir(save_direc)
    except:
        print('yep!')
    #initialize for the chogi_subjects
    mbep = ext_epochs('ALPHA/sta1', MBlist, 0)
    mfep = ext_epochs('ALPHA/sta1', MFlist, 0)
    mbep2 = ext_epochs('ALPHA/sta2', MBlist2, 0)
    mfep2 = ext_epochs('ALPHA/sta2', MFlist2, 0)

    for i in range(np.asarray(MBlist).shape[0] - 1):
        print('-' * 10, ' : [', i + 1, '|', np.array(MBlist).shape[0], ']', '-' * 10)
        mbep = np.concatenate((mbep, ext_epochs('ALPHA/sta1', MBlist, i + 1)), 2)
        mfep = np.concatenate((mfep, ext_epochs('ALPHA/sta1', MFlist, i + 1)), 2)
        mbep2 = np.concatenate((mbep2, ext_epochs('ALPHA/sta2', MBlist2, i + 1)), 2)
        mfep2 = np.concatenate((mfep2, ext_epochs('ALPHA/sta2', MFlist2, i + 1)), 2)
    for cc in range(64):
        try:
            os.mkdir(save_direc + '/channel_{0}'.format(cc))
        except:
            print('yep!')
        epo_mb = channel_selection(mbep,cc)
        epo_mf = channel_selection(mfep,cc)
        epo_mb2 = channel_selection(mbep2, cc)
        epo_mf2 = channel_selection(mfep2, cc)

        # mb as 1 and mf as -1
        epo_mb = np.reshape(epo_mb,(1,epo_mb.shape[0],epo_mb.shape[1]))
        epo_mf = np.reshape(epo_mf,(1,epo_mf.shape[0],epo_mf.shape[1]))
        mbraw = ext_erp(epo_mb)
        mfraw = ext_erp(epo_mf)

        epo_mb2 = np.reshape(epo_mb2,(1,epo_mb2.shape[0],epo_mb2.shape[1]))
        epo_mf2 = np.reshape(epo_mf2,(1,epo_mf2.shape[0],epo_mf2.shape[1]))
        mbraw2 = ext_erp(epo_mb2)
        mfraw2 = ext_erp(epo_mf2)

        mblbl = np.ones((mbraw.shape[0]))
        mflbl = np.ones((mfraw.shape[0]))*(-1)

        mblbl2 = np.ones((mbraw2.shape[0]))
        mflbl2 = np.ones((mfraw2.shape[0]))*(-1)

        full_data = np.concatenate((mbraw,mfraw),0)
        full_label = np.concatenate((mblbl,mflbl),0)

        full_data2 = np.concatenate((mbraw2,mfraw2),0)
        full_label2 = np.concatenate((mblbl2,mflbl2),0)

        accuracy_tw = []
        for ii in range(mbraw.shape[1]):
            print('1. Feature selection')
            start_time = time.time()
            data = full_data[:,ii,:]
            print("--- %s seconds ---" % (time.time() - start_time))

            data_folded = KFold(n_splits=10, shuffle=True)
            data_folded.get_n_splits(data)

            accuracy_list = []
            print('2. SVM')
            start_time = time.time()
            for train_index, test_index in data_folded.split(data):
                X_train, X_test = data[train_index], data[test_index]
                y_train, y_test = full_label[train_index], full_label[test_index]
                svm_model = SVM()
                svm_model.train(X_train,y_train)

                accuracy_list.append(svm_model.accuracy_score(X_test, y_test))
            print("--- %s seconds ---" % (time.time() - start_time))

            print('1. Feature selection')
            start_time = time.time()
            data = full_data2[:,ii,:]
            print("--- %s seconds ---" % (time.time() - start_time))

            data_folded = KFold(n_splits=10, shuffle=True)
            data_folded.get_n_splits(data)

            accuracy_list = []
            print('2. SVM')
            start_time = time.time()
            for train_index, test_index in data_folded.split(data):
                X_train, X_test = data[train_index], data[test_index]
                y_train, y_test = full_label2[train_index], full_label2[test_index]
                svm_model = SVM()
                svm_model.train(X_train,y_train)

                accuracy_list.append(svm_model.accuracy_score(X_test, y_test))
            print("--- %s seconds ---" % (time.time() - start_time))

            accuracy_tw .append(accuracy_list)
        print('3. saving')
        start_time = time.time()
        filename1 = save_direc + '/channel_{0}'.format(cc) + '/acc'
        np.save(filename1,accuracy_tw)
        print("--- %s seconds ---" % (time.time() - start_time))
