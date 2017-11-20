from sklearn import svm
from sklearn.feature_selection import mutual_info_classif as MINF
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import numpy as np
import scipy
import scipy.io
import fnmatch
import os
from scipy import signal
import matplotlib.pyplot as plt
import time

def ext_files(direc_cond, postfix):
    # direc_cond refers to the directory and prefix
    lom = [] # list of matrix.
    for file in os.listdir(direc_cond):
        if fnmatch.fnmatch(file, '*' + postfix + '.mat'):
            lom.append(file)
    lom.sort()
    return lom

def ext_epochs(direc_cond, lom, indexforlom):
    # first, you have to use ext_files to extract list of files. => lom : list of mat-file
    lom.sort()
    epochs = scipy.io.loadmat(direc_cond +'/'+ lom[indexforlom] )['epochs']
    return epochs
