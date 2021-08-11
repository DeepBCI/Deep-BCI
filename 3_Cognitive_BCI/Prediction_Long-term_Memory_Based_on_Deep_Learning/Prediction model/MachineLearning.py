import os

import numpy as np

import matplotlib.pyplot as plt
import DataPreprocessing as DP

from sklearn.svm import SVC
from sklearn.metrics import cohen_kappa_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

# file path
datapath = os.getcwd()

# Picture memory
datatype = './PM_PSD_24H/'
# Location memory
# datatype='./LM_PSD_24H/'

# 1D
x_datapath = 'x_data/'
y_datapath = 'y_data/'

x_filepath = datapath + datatype + x_datapath
y_filepath = datapath + datatype + y_datapath

x_filelist = os.listdir(x_filepath)
y_filelist = os.listdir(y_filepath)

# sort
x_filelist.sort()
y_filelist.sort()

# file load
x_data = []
y_data = []


for i in range(len(x_filelist)):
    x_data.append(DP.preprocess_data(x_filepath, x_filelist[i]))
    y_data.append(DP.load_header(y_filepath, y_filelist[i]))

acc_mean = []
kappa_mean = []
conf_matrix_all = []
for fold in range(len(x_data)):
    [trainx, trainy, testx, testy] = DP.split_train_test(x_data, y_data, fold)

    trainx = np.transpose(trainx, (1, 0))
    testx = np.transpose(testx, (1, 0))

    # SVM
    clf = SVC(kernel='rbf')
    # clf = SVC(kernel='rbf', class_weight='balanced')

    # RandomForset
    # clf = RandomForestClassifier(class_weight='balanced')

    clf.fit(trainx, np.ravel(trainy))

    clf_predictions = clf.predict(testx)
    kappa = cohen_kappa_score(np.ravel(testy), clf_predictions)
    acc = accuracy_score(np.ravel(testy), clf_predictions) * 100
    print('Accuracy: %.3f, Kappa: %.3f' % (acc, kappa))

    # confusion matrix
    conf_matrix = confusion_matrix(np.ravel(testy), clf_predictions)

    acc_mean.append(acc)
    kappa_mean.append(kappa)

    # confusion matrix
    conf_matrix_all.append(conf_matrix)

Acc_avg = np.mean(acc_mean)
Acc_std = np.std(acc_mean)

kappa_avg = np.mean(kappa_mean)
kappa_std = np.std(kappa_mean)


# confusion matrix
conf_max = sum(conf_matrix_all)
print(conf_max)
print(conf_max.astype('float') / conf_max.sum(1)[:, None])

print('Acc_mean: %.3f(%.3f), Kappa_mean: %.3f(%.3f)' % (Acc_avg, Acc_std, kappa_avg, kappa_std))