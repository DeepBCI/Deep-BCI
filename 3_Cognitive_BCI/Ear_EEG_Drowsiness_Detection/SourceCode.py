# -*- coding: utf-8 -*-
"""
Created on Fri 15th May 2020

@ Name: Chang-Hee Han (Ph.D)
@ Position: Postdoctoral Researcher
@ Affiliation: Machine Learning Group, Technical University of Berlin 
@ Address: Marchstr. 23, D-10587 Berlin, Germany
@ Email: zeros8706@naver.com / chang-hee.han@campus.tu-berlin.de
@ HP: +49 17 6657 96308
@ Google Scholar: https://goo.gl/Bu7vde
@ Homepage: https://zeros8706.wixsite.com/changheehan/

"""


###############################################################################
### (1) Enable logging ########################################################
###############################################################################
#import logging
#import importlib
#importlib.reload(logging) # see https://stackoverflow.com/a/21475297/1469195
#log = logging.getLogger()
#log.setLevel('INFO')
#import sys
#logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s',
#                     level=logging.INFO, stream=sys.stdout)

Timesegment = 10 # Period of a trial (default = 10 sec)
Fs = 200 # Sampling rate (default = 200 Hz)

###############################################################################
### (2) Load data #############################################################
###############################################################################
# set up data path
import os
default_path = 'D:/Changhee/4_EarEEG EOandEC/CNNData_Onset_200508' # EEG data path
result_path = 'D:/Changhee/4_EarEEG EOandEC/CNNData_Onset_200508/Result_1s50%/50' # Result save path
os.chdir(default_path)

#from tqdm import tqdm
import warnings
warnings.simplefilter('ignore')

# load raw data and labels
import numpy as np
for temp_subNum in range(1, 31): # change according to the number of subjects (default = 30)
    print(temp_subNum)

    from scipy import io
    raw_data = io.loadmat('sub_' + str(temp_subNum) + '_Day1_TR_200508.mat') # load raw EEG data
    raw_labels = io.loadmat('sub_' + str(temp_subNum) + '_Day1_200508_ylabel.mat') # Load trial labels (0 or 1)
    FeatVect = raw_data['rawData_TR']
    FeatVect = FeatVect[:, :, 0:(Timesegment*Fs)]
    y_labels = raw_labels['y_label']
    del raw_data, raw_labels
    
    ###########################################################################
    ### (3) Preprocessing #####################################################
    ###########################################################################
    from braindecode.datautil.signalproc import lowpass_cnt, highpass_cnt, exponential_running_standardize
    for ii in range(0, 60): # change according to the number of trials (default = 60)
        # 1. Data reconstruction
        temp_data = FeatVect[ii, :, :]
        temp_data = temp_data.transpose()
        # 2. Lowpass filtering
        lowpassed_data = lowpass_cnt(temp_data, 13, 200, filt_order=3)
        # 3. Highpass filtering
        bandpassed_data = highpass_cnt(lowpassed_data, 8, 200, filt_order=3)
        # 4. Exponential running standardization
        ExpRunStand_data = exponential_running_standardize(bandpassed_data, factor_new=0.001, init_block_size=None, eps=0.0001)
        # 5. Renewal preprocessed data
        ExpRunStand_data = ExpRunStand_data.transpose()
        FeatVect[ii, :, :] = ExpRunStand_data
        del temp_data, lowpassed_data, bandpassed_data, ExpRunStand_data
    
    ###########################################################################
    ### (3) Convert data to braindecode format ################################
    ###########################################################################
    # pytorch expects float 32 for input and int64 for labels.
    X = (FeatVect).astype(np.float32)
    y = (y_labels).astype(np.int64)
    y = np.delete(y, [60], None)
    del FeatVect, y_labels
    
    from braindecode.datautil.signal_target import SignalAndTarget
    from braindecode.datautil.splitters import split_into_two_sets
    
    train_index_1 = list(range(0, 25))
    train_index_2 = list(range(30, 55)) 
    train_index = np.append(train_index_1, train_index_2)
    del train_index_1, train_index_2
    
    test_index_1 = list(range(25, 30))
    test_index_2 = list(range(55, 60)) 
    test_index = np.append(test_index_1, test_index_2)
    del test_index_1, test_index_2
        
    TR_size = train_index.shape[0] # Training trial size (default = 50)
    TW_size = 1*Fs # Time window size
    ST_size = 100 # Opveral size
    X_train = np.zeros((TR_size*19, X.shape[1], TW_size))
    X_train = X_train.astype(np.float32)
    y_train = np.zeros((TR_size*19))
    y_train = y_train.astype(np.int64)
    
    for ii in range(1, 20):
        X_train[0+(ii-1)*TR_size:(TR_size)+(ii-1)*TR_size, :, 0:TW_size] = X[train_index, :, 0+(ii-1)*ST_size:TW_size+(ii-1)*ST_size]
        y_train[0+(ii-1)*TR_size:(TR_size)+(ii-1)*TR_size] = y[train_index]
    
    train_set = SignalAndTarget(X_train, y_train)
    train_set, valid_set = split_into_two_sets(train_set, first_set_fraction=0.8)
    
    ###########################################################################
    ### (4) Create the model ##################################################
    ###########################################################################
    from braindecode.models.shallow_fbcsp import ShallowFBCSPNet
    from braindecode.torch_ext.util import set_random_seeds
    
    # Set if you want to use GPU
    # You can also use torch.cuda.is_available() to determine if cuda is available on your machine.
    cuda = True
    set_random_seeds(seed=20170629, cuda=cuda)
    n_classes = 2
    in_chans = train_set.X.shape[1]
    
    model = ShallowFBCSPNet(in_chans=in_chans, n_classes=n_classes, input_time_length=None, n_filters_time=40,
                            filter_time_length=25, n_filters_spat=40, pool_time_length=30,
                            pool_time_stride=4, final_conv_length=12,
                            pool_mode='mean', split_first_layer=True,
                            batch_norm=True, batch_norm_alpha=0.1, drop_prob=0.5)
    
    if cuda:
        model.cuda()
        
    ###########################################################################
    ### (5) Create cropped iterator ###########################################
    ###########################################################################
    from braindecode.torch_ext.optimizers import AdamW
    import torch.nn.functional as F
    #optimizer = AdamW(model.parameters(), lr=1*0.01, weight_decay=0.5*0.001) # these are good values for the deep model
    optimizer = AdamW(model.parameters(), lr=0.0625 * 0.01, weight_decay=0)
    model.compile(loss=F.nll_loss, optimizer=optimizer,  iterator_seed=1, cropped=True)
    
    ###########################################################################
    ### (6) Run the training ##################################################
    ###########################################################################
    #input_time_length = 1*Fs
    input_time_length = 200
    model.fit(train_set.X, train_set.y, epochs=250, batch_size=64, scheduler='cosine',
              input_time_length=input_time_length, remember_best_column='valid_misclass',
              validation_data=(valid_set.X, valid_set.y),)
    
    # model.epochs_df
    
    ###########################################################################
    ### (7) Evaluation ########################################################
    ###########################################################################
    MovWind_Start = np.arange(0,1900,100)
    MovWind_End = MovWind_Start + 200 
    PredictedLabels = np.zeros((test_index.shape[0], 19))
    for temp_movSteps in range(0, 19):
        X_test = X[test_index, :, MovWind_Start[temp_movSteps]:MovWind_End[temp_movSteps]]
        y_test = y[test_index]
        test_set = SignalAndTarget(X_test, y_test)
        
        #scores = model.evaluate(test_set.X, test_set.y)
        
        #Acc = (1 - scores['misclass']) * 100
        
        PredictedLabels[:, temp_movSteps] = model.predict_classes(test_set.X)
        
    del optimizer, model, input_time_length, in_chans, n_classes, X_train, X_test, y_train, y_test, train_set, valid_set, test_set
    
    os.chdir(result_path)
    #io.savemat(file_name='sub_' + str(temp_subNum) + '_Day1_Result_labels_200508.mat', mdict={'PredictedLabels': (PredictedLabels)})
    os.chdir(default_path)
    
    del PredictedLabels
