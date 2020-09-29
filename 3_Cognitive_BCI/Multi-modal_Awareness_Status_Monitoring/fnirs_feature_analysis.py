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
