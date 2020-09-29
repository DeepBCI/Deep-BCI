# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 16:21:55 2020

@author: raxus
"""
# load data
import pandas as pd
import numpy as np

df_nirs=pd.read_csv("C:\\Users\\raxus\\Documents\\MATLAB\\matlab 코드 파일\\sigrand627.csv", encoding='utf-8', header=None)
f_names=['Class label']
for feature in ['mean', 'var', 'skew', 'kurt','peak','time-to-peak','slope','AUC','RMS']:
    for i in range(1,10):
        f_names.append(feature+'_%d'%i)
df_nirs.columns = f_names

print(df_nirs.shape)


# split data
from sklearn.model_selection import train_test_split
X, y = df_nirs.iloc[:,1:], df_nirs.iloc[:,0]
X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)


# feature scaling
from sklearn.preprocessing import MinMaxScaler

minmax = MinMaxScaler()
X_train = minmax.fit_transform(X_train)
X_test = minmax.transform(X_test)

# demension reduction
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2)
X_train_tsne = tsne.fit_transform(X_train)