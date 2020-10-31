import numpy as np
import scipy.io as spio
import os


from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers import Dropout

# 1. 데이터셋 생성하기
Temp_mat1 = spio.loadmat('./fNIRS_dataset_3Hb_train.mat')
Temp_mat2 = spio.loadmat('./fNIRS_dataset_3Hb_val.mat')
Temp_mat3 = spio.loadmat('./fNIRS_dataset_3Hb_test.mat')

x_train = Temp_mat1['x_train']
y_train = Temp_mat1['y_train']

x_test = Temp_mat2['x_val']
y_test = Temp_mat2['y_val']

x_val = Temp_mat3['x_test']
y_val = Temp_mat3['y_test']

x_train = x_train.reshape(977, 1200, 38, 3).astype('float32')
x_test = x_test.reshape(210, 1200, 38, 3).astype('float32')
x_val = x_val.reshape(209, 1200, 38, 3).astype('float32')

# 2. 모델 구성하기
from keras.layers.normalization import BatchNormalization

model = Sequential()
model.add(Conv2D(128, (5, 5), padding='same', activation='relu', input_shape=(1200, 38, 3)))
# model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(4, 1)))
model.add(Dropout(0.25))
model.add(Conv2D(128, (5, 5), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(4, 1)))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 1)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 3. 모델 학습시키기
hist = model.fit(x_train, y_train, epochs=20, batch_size=40, validation_data=(x_val, y_val))

# 4. 학습과정 살펴보기
%matplotlib inline
import matplotlib.pyplot as plt

fig, loss_ax = plt.subplots()

acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'], 'y', label='train loss')
loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')
loss_ax.set_ylim([0.0, 1])

acc_ax.plot(hist.history['acc'], 'b', label='train acc')
acc_ax.plot(hist.history['val_acc'], 'g', label='val acc')
acc_ax.set_ylim([0.0, 1.0])

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuray')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')

plt.show()