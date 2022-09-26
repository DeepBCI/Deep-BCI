# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 11:15:03 2021

@author: yelee
"""




import tensorflow
import gym
import scipy.io as sio
import os
import numpy as np

import datetime
import tensorflow as tf
import keras
from keras.callbacks import TensorBoard

from keras.models import Sequential,Model
from keras.layers import Input, Dense, Activation, Dropout, Reshape,MaxPooling2D, Flatten,concatenate
from keras.optimizers import Adam, SGD, RMSprop, Adagrad, Adadelta, Adamax, Nadam
from keras import metrics, initializers
from keras.layers import TimeDistributed,Conv2D,Conv1D,LSTM,SeparableConv2D,DepthwiseConv2D,BatchNormalization

#from keras.utils.training_utils import multi_gpu_model
from keras.layers import convolutional_recurrent
from sklearn import svm
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from keras.utils import multi_gpu_model
import keras.backend.tensorflow_backend as K
from sklearn.model_selection import train_test_split,KFold,StratifiedKFold
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import seaborn as sns

# %% data load
dire = 'A:/InnerSpeech_DATA_shlee/B2_epo_300tr/'

totalSub = 9
# TP7: 50, C5: 46, T7: 11
chan_bro_wer = [3,7,11,17,33,36,42,46,50,55] # 0 기준으로 바꾼거
# chan_bro_wer = list(range(0,64))
# perception이랑 rest 는 sess 1으로 하기

session = 'imagined' # imagined, overt, perception, rest
X = []
Y_ = []
for subNum in range(1,totalSub+1):
    y_ = np.zeros((300,1)) # production: 1200, perception: 300
    data = sio.loadmat(os.path.join(dire, 'sub' + '%d' % subNum + '_%s' %session + '.mat'))  # read training data set
    
    epo = data['epo']
    x_ = np.expand_dims(np.transpose(np.array(epo[0,0]['x']),[2,0,1]),axis=3)
    X.extend(x_[:,:,chan_bro_wer,:])
    # X.extend(np.expand_dims(x_[:,:,chan_bro_wer,:],axis=2))
    
    y_[:,0] = subNum-1
    Y_.extend(y_)
  
X = np.array(X)
Y_ = np.squeeze(np.array(Y_))

# %% to category for deep learning
Y = tf.keras.utils.to_categorical(Y_, num_classes=totalSub)

kf = StratifiedKFold(n_splits=5,shuffle=True, random_state=1)

split_index = list(kf.split(X,Y_))

# %% model define

initializer = initializers.glorot_normal(seed=None)  # glorot_normal, random_normal

def CNN_model(input_cnn):
    
    fs = 250
    # temporal features  int(fs/2)
    x = Conv2D(16, (int(fs/2),1), padding="valid", activation="relu",kernel_initializer=initializer)(input_cnn)
    x = BatchNormalization()(x)
    # x = MaxPooling2D(pool_size=(2,1))(x)
    
    
    
    x = SeparableConv2D(8, (25, 1), padding="valid",activation="relu",kernel_initializer=initializer)(x)
    x = SeparableConv2D(16, (15, 1), padding="valid",activation="relu",kernel_initializer=initializer)(x)
    x = SeparableConv2D(32, (6, 1), padding="valid",activation="relu",kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2,1))(x)
    
    # (3,1)
    x = Conv2D(64, (3,1), padding="valid", activation="relu",kernel_initializer=initializer)(x)
    x = MaxPooling2D(pool_size=(2,1))(x)
    x = BatchNormalization()(x)
    
    # channel wise
    x = Conv2D(16, (1,input_cnn.shape[2]),padding="valid", activation="relu",kernel_initializer=initializer)(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dense(totalSub,activation="softmax")(x)
    
    x = Model(inputs=input_cnn, outputs=x)
    
    x.summary()
    
    return x

input_cnn = Input(shape=(501,1,1))

with K.tf.device('/gpu:1'):
    model = CNN_model(input_cnn)
# %%
lr = 0.5
epochs = 1000
batch_size = 240
lr_decay = 0.00001

dire_logs='A:/InnerSpeech_DATA_shlee/results_index/chanSelect/'


model_list = []
train_history_list = []
eval_metric_list=[]
pred_te_list = []
pred_tr_list = []
conf_tr_list = []
conf_te_list = []
tr_idx_list = []
te_idx_list = []
conf_ch_list = []
acc_ch_list = []
acc_te=[]
intermediates_tsne2d_list=[];


with K.tf.device('/gpu:1'):
        # model = CNN_model(input_cnn) 
    for ch in range(0,len(chan_bro_wer)):
        
        chan = chan_bro_wer[ch]
        X_ch = np.expand_dims(X[:,:,ch,:],axis=2)

        
        
        for k in range(0,len(split_index)):
            
            train_index, test_index = split_index[k]
            
            print('chan: ' + str(chan))
            print(str(k) + ' th fold')
            
            tr_idx_list.append(train_index)
            te_idx_list.append(test_index)
            trX, teX = X_ch[train_index], X_ch[test_index]
            trY, teY = Y[train_index], Y[test_index]
            
           
            earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=20)
            model = CNN_model(input_cnn)
        
            
            #multi-gpu
            # model = multi_gpu_model(model, gpus=2)
            
            # For a mean squared error regression problem
            model.compile(loss='squared_hinge', 
                          optimizer=Adadelta(lr=lr, decay = lr_decay), #, decay = lr_decay 
                          metrics=['accuracy'])
            
            # log_dir = dire_logs + "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            # log_dir = dire_logs + "logs"
            
            # tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1) #, histogram_freq=1
            
            checkpoint_filepath = dire_logs+'checkpoint/check_%s_ch%d_%d' %(session, chan, k)
            model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_filepath,
                save_weights_only=True,
                monitor='val_accuracy',
                mode='max',
                save_best_only=True)
        
            # Train the model, iterating on the data in batches of 32 samples
            train_history = model.fit(trX, trY, 
                                      validation_data = (teX,teY), 
                                      epochs=epochs, verbose=1, batch_size=batch_size,
                                      callbacks=[model_checkpoint_callback, earlystop_callback])
                                      # callbacks=[tensorboard_callback])
            train_history_list.append(train_history)
            
            model.load_weights(checkpoint_filepath)
            
            eval_metric = model.evaluate(teX,teY)#,batch_size=batch_size)
            acc_te.append(eval_metric[1])
            
            
            eval_metric_list.append(eval_metric)
            
            pred_te = model.predict(teX)
            pred_tr = model.predict(trX)
            pred_te_list.append(pred_te)
            pred_tr_list.append(pred_tr)
            
            print('Train score: ' + str(r2_score(trY.flatten(), pred_tr.flatten())))
            print('Test score: ' + str(r2_score(teY.flatten(), pred_te.flatten())))
            print("acc : " , eval_metric[1], " , c : " , eval_metric[0])
            
            model_list.append(model)
            model.save(dire_logs + 'models/model_%s_ch%d_%d' %(session, chan,k))
        
        
        # % quick check
            train_history.history.keys()
            
            val_loss_lst = train_history.history['val_loss']
            train_loss_lst = train_history.history['loss']
            
            plt.figure(figsize=(12, 4))
            plt.plot(range(0, len(train_loss_lst)), train_loss_lst, label='train_loss')
            plt.plot(range(0, len(val_loss_lst)), val_loss_lst, label='val_loss')
            plt.legend()
            plt.title('Loss Function')
            plt.savefig(dire_logs+'loss/loss_%s_ch%d_%d.png' %(session, chan, k))
            #plt.savefig(dire + 'cnn_train_val_score.png')
            plt.show()
            
            
            conf_tr = confusion_matrix(np.argmax(trY, axis=1), np.argmax(pred_tr, axis=1))
            conf_te = confusion_matrix(np.argmax(teY, axis=1), np.argmax(pred_te, axis=1))
            
            conf_tr_list.append(conf_tr)
            conf_te_list.append(conf_te)
            
            
            plt.figure(figsize=(20,20))
            sns.set(font_scale=6) # font size 2
            ax = sns.heatmap(conf_te, annot=True, fmt='d', cmap='Blues',
                              linewidths=1,xticklabels=False,yticklabels=False,
                              cbar=False)
            plt.savefig(dire_logs+'fig/2_CM_%s_te_%d.svg' %(session, k))
            
      
        
            # t-SNE
            plt_X = X
            plt_Y = Y
            layer_of_interest=6
            intermediate_tensor_function = K.function([model.layers[0].input],[model.layers[layer_of_interest].output])
            intermediate_tensor = intermediate_tensor_function([np.expand_dims(plt_X[0,:],axis=0)])[0]
            
            color_list = ['#000000', '#980000','#ff0000','#ff9900','#ffff00',
                          '#00ff00','#00ffff','#4a86e8','#0000ff','#9900ff',
                          '#ff00ff','#7f6000','#0c343d']
            intermediates = []
            color_intermediates = []
            for i in range(len(plt_X)):
                output_class = np.argmax(plt_Y[i,:])
                intermediate_tensor = intermediate_tensor_function([np.expand_dims(plt_X[i,:],axis=0)])[0]
                intermediates.append(np.ravel(intermediate_tensor))
                color_intermediates.append(color_list[output_class])
            
            
        
            tsne2d = TSNE(n_components=2, random_state=0)
            intermediates_tsne2d = tsne2d.fit_transform(intermediates)
            
            plt.figure(figsize=(8,8))
            plt.scatter(intermediates_tsne2d[:,0],intermediates_tsne2d[:,1], color=color_intermediates)
            plt.savefig(dire_logs+'fig/2_TSNE_%s_%d.svg' %(session, k))
            plt.show()
            
            intermediates_tsne2d_list.append(intermediates_tsne2d)
        

        # saved variables
    
        # fig 2 (a) Confusion matrix
        conf_te_avg = np.mean(np.array(conf_te_list),axis=0)
        
        conf_ch_list.append(conf_te_avg)
        plt.imshow(conf_te_avg,cmap=plt.cm.Blues)
        plt.xlabel("Predicted labels")
        plt.ylabel("True labels")
        plt.xticks([], [])
        plt.yticks([], [])
        plt.title('Normalized confusion matrix of test results')
        plt.colorbar(ticks=np.arange(0.,1.1,0.2))
        plt.savefig(dire_logs+'fig/2_CM_%s_te_avg.svg' % session)
        plt.show()
    
        # fig 2 (b) T-SNE
        label = ['Subject 1','Subject2','Subject3','Subject4','Subject5','Subject6','Subject7','Subject8','Subject9']
        tSNE_avg = np.mean(np.array(intermediates_tsne2d_list),axis=0)
        plt.figure(figsize=(8,8))
        plt.scatter(tSNE_avg[:,0],tSNE_avg[:,1], color=color_intermediates,label=label)
        plt.legend()
        # plt.savefig(dire_logs+'fig/2_TSNE_%s_avg.svg' %session)
    
        plt.show()
        
        # fig 3 values avg + std
        eval_metric_avg = np.mean(np.array(eval_metric_list),axis=0)[1]
        eval_metric_std = np.std(np.array(eval_metric_list),axis=0)[1]
        
        with open(dire_logs+'eval_metric_%s_ch%d.txt' % (session,chan), 'w') as f:
            # f.write('average: %f +- %f\n' %(eval_metric_avg, eval_metric_std))
            f.write('accuracy: ' + str(np.array(eval_metric_list)[:,1]))
         
        acc_ch_list.append(eval_metric_avg)
        
# %% New User
newX = sio.loadmat(os.path.join(dire,'newUser.mat'))
   
newX_ch = np.expand_dims(newX[:,:,ch,:],axis=2)

pred_new = model.predict(newX)

if pred_new in range(totalSub):
    print('This User can access')
else:
    print('Deny Access')

