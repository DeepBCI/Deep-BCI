from sklearn import svm
from sklearn.feature_selection import mutual_info_classif as MINF
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score as acc_func
import numpy as np
import scipy
import scipy.io
import fnmatch
import os
from scipy import signal
import time

class SVM:
    def __init__(self):
        self.model = svm.SVC()

    def train(self,x_train,y_train):
        return self.model.fit(x_train,y_train)

    def predict(self,x_test):
        estimation = self.model.predict(x_test)
        return estimation

    def accuracy_score(self, x_test, y_test):
        estimation = self.model.predict(x_test)
        return acc_func(estimation,y_test)

