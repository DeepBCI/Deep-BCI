# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 23:33:11 2019

@author: choi
"""

import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Dense
from sklearn.model_selection import KFold
from keras.callbacks import ModelCheckpoint

binary_dataset = pd.read_csv("E:/Project/DeepLearning_test/EC_Temporal_8ch.csv", header = None)
#binary_dataset = pd.read_csv("E:/Project/2019/WISET/Deeplearning/earECG.csv", header = None)
z= 0
acc = np.zeros([30,4])

for ch in [0,1]:
  
    X = binary_dataset.values[:, 4:-1]
            
    n = 0
    m = 0
    for sess in range(len(X)//30):
        
        Y = np.zeros([len(X),1])
        for sub in range(n,n+(len(X)//30)):
            Y[sub,:] = 1
            
        from random import shuffle
        ind_list = [i for i in range(len(X))]
        shuffle(ind_list)
        
        X = X[ind_list, :]
        Y = Y[ind_list, ]
            
        from sklearn.model_selection import train_test_split
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state =42)
        
        
        from keras.models import Sequential
        from keras.layers import Dense, Flatten, Dropout, BatchNormalization
        from keras.layers.convolutional import Conv1D, MaxPooling1D
            
        X_tr = np.expand_dims(X_train, axis = 2)
        #x_tr = np.stack([X_train, X_train, X_train, X_train, X_train, X_train],axis = 1)
        
        n_timestep, n_signal = X_tr.shape[1], X_tr.shape[2]
        #n_timestep, n_signal = x_tr.shape[1], x_tr.shape[2]
        
        # CV
        import matplotlib.pyplot as plt
        
        def plot_acc_lost(history, fold):
            plt.figure(figsize=(6, 4)) 
            plt.plot(history.history['acc'], 'r', label = 'Accuracy of Training Data')
            plt.plot(history.history['val_acc'], 'b', label = 'Accuracy of Validation Data')
            plt.plot(history.history['loss'], 'r--', label = 'Loss of Training Data') 
            plt.plot(history.history['val_loss'], 'b--', label = 'Loss of Validation Data')
            plt.title('Model Accuracy and Loss of fold') 
            plt.ylabel('Accuracy and Loss') 
            plt.xlabel('Training Epoch') 
            plt.ylim(0) 
            plt.legend() 
            plt.show() 
        
        # 5-fold cross-validation example
        kf = KFold(n_splits=5, shuffle = True)
        accuracy_list = []
        loss_list = []
        fold = 0
        
        for fold in range(0,1):
            print("====================== Fold =======================")
            print(fold)
            print("===================================================")
                
            model = Sequential()
            model.add(Conv1D(filters= 32, kernel_size = 3 ,strides = 2, activation = 'relu', input_shape=(n_timestep, n_signal)))
            #model.add(Conv1D(filters= 2, kernel_size = 3, strides = 1, activation = 'relu', input_shape=(n_timestep, n_signal)))
            model.add(BatchNormalization())
            model.add(MaxPooling1D(pool_size=1))
            model.add(Flatten())
            model.add(Dropout(0.3))
            model.add(Dense(units= 20, kernel_initializer = 'uniform', activation = 'relu'))
            model.add(Dense(units=1, kernel_initializer = 'uniform', activation = 'sigmoid'))
            model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics =['accuracy'])
            model.summary()
            
            history = model.fit(X_tr, Y_train, epochs=100, batch_size=30, verbose=1)
                
            X_te = np.expand_dims(X_test, axis=2)
            _, acc[m,:] = model.evaluate(X_te, Y_test, batch_size=25)
            print(acc)
            m = m+1
         
        n = n+(len(X)//30)
  