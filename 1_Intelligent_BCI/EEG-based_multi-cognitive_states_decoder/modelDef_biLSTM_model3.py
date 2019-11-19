from sklearn import svm
from sklearn.feature_selection import mutual_info_classif as MINF
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score as acc_func
import scipy
import scipy.io
import fnmatch
import os
from scipy import signal
import matplotlib.pyplot as plt
import time
import numpy as np
import tensorflow as tf
"""
class CNN31D :
    # origianl input data size : 61x61x64(frequency x time point x channel number) to 1D image
    # reference : A novel deep learning approach for classification of EEG motor imagerty signals, Yousef Razael Tabar and Ugur Halici, 2017, Journal of Neural Engineering
    #
    def __init__(self, meta_json, save_direc, sub, cv):
        input_layer = 3
        self.meta_json = meta_json
        self.ccc = 0
        self.save_dir=save_direc
        self.sub = sub
        self.cv = cv

        return

    def initialize_variables(self):
        # reshape x_train to 2D image
        # For this case, use DNN with 5 channels only : E6(Fz), E10(Fp1), E5(Fp2), E18(F7), E58(F8)
        # x_train = np.zeros((100,64,61,61))
        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

        channel_index = (4, 5, 9, 17, 57)

        # conv network
        self.x_in = tf.placeholder(shape = [None, 61*5, 61, 1], dtype=tf.float32)
        self.y_ = tf.placeholder(dtype=tf.float32, shape = [None, 2])

        w1 = tf.Variable(tf.truncated_normal(shape = (61*5,7,1,60) , stddev=0.1))
        bias1 = tf.Variable(tf.constant(0.1, shape = [60]))
        conv1 = tf.nn.conv2d(self.x_in, w1, strides = [1,1,1,1], padding="VALID")
        conv1_out = tf.nn.relu(conv1 + bias1)
        pool1 = tf.nn.max_pool(conv1_out, ksize= [1,1,5,1], strides=[1,1,5,1], padding='SAME')

        # This model cannot be structurally deep since this model applied 1d convolution
        self.weight_ = tf.Variable(tf.zeros((660,2),dtype=tf.float32),dtype=tf.float32)
        self.bias_ = tf.Variable(tf.zeros((2),dtype=tf.float32),dtype=tf.float32)
        self.pool1_flat = tf.reshape(pool1, shape=[-1, 660])
        self.y = tf.add(tf.matmul(self.pool1_flat, self.weight_), self.bias_)

        # cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.y, labels=self.y_))

        self.train_step = tf.train.AdamOptimizer(learning_rate=self.meta_json["LEARNING_RATE"]).minimize(self.loss)

        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()

        return


    def restoring_variables(self):
        som = [f for f in os.listdir(self.meta_json["MODEL_NAME"] + '-'  + self.meta_json["STARTED_DATESTRING"] + '/.') if f.lower().endswith('iter.ckpt')]
        soi = []
        for i in som:
            soi.append(int(i.split('.')[0][5:-4]))
        self.saver.restore(self.sess, self.meta_json["MODEL_NAME"] + '-'  + self.meta_json["STARTED_DATESTRING"] + "/model.ckpt")
        self.ccc = 100

        return


    def train(self, x_train, y_train, x_test, y_test):

        if self.meta_json["rest"] is True:
            self.initialize_variables()
            self.restoring_variables()
        else:
            self.initialize_variables()

        channel_index = (4, 5, 9, 17, 57)
        x_train_ = x_train[:,channel_index,:,:]
        x_tt = np.zeros((x_train_.shape[0], 61*5, 61, 1))
        for i in range(x_train_.shape[3]):
            x_tt [:,:,i,:] = x_train_[:,:,:,i].reshape((-1,x_train_.shape[2]*x_train_.shape[1], 1))

        y_mod = np.zeros((y_train.shape[0],2), dtype=np.int16)
        y_train = y_train.astype(np.float32)
        for i in range(y_train.shape[0]):
            y_mod[i, int(((y_train[i]) + 1)/2)] = 1
        # y = tf.Variable(y_mod,dtype=tf.float32)
        y = y_mod.astype(np.float32)


        loss_ = self.loss
        loss_hist = tf.summary.scalar('loss', loss_)
        merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(
            './' + self.meta_json["MODEL_NAME"] + self.meta_json["STARTED_DATESTRING"] + self.meta_json["LOGDIR_ROOT"],
            self.sess.graph)

        # x_tt = x_train.reshape((x_train.shape[0], -1) ).astype(np.float32)
        # x_tt = tf.Variable( tf.cast( tf.reshape(x_train, (x_train.shape[0], -1) ), dtype=tf.float32) )
        y_mod = np.zeros((y_train.shape[0],2), dtype=np.int16)
        y_train = y_train.astype(np.float32)
        for i in range(y_train.shape[0]):
            y_mod[i, int(((y_train[i]) + 1)/2)] = 1
        # y = tf.Variable(y_mod,dtype=tf.float32)
        y = y_mod.astype(np.float32)
        for i in range(int(self.meta_json["NUM_STEPS"])):
            self.sess.run(self.train_step, feed_dict = {self.x_in: x_tt, self.y_: y})
            if i%(1000) == 0:
                self.ccc += 1
                print('[{0}/{1}] : {2} -- loss : {3}'.format(self.ccc , int(self.meta_json["NUM_STEPS"]/1000) , self.accuracy_score( x_test, y_test), self.sess.run(self.loss, feed_dict = {self.x_in: x_tt, self.y_: y}) ) )
                try:
                    save_path = self.saver.save(self.sess, self.meta_json["MODEL_NAME"] + '-' + self.meta_json[
                        "STARTED_DATESTRING"] + "/model.ckpt")
                    save_path = self.saver.save(self.sess, self.meta_json["MODEL_NAME"] + '-' + self.meta_json[
                        "STARTED_DATESTRING"] + "/model" + str(i) + "iter.ckpt")
                    summary = self.sess.run(merged, feed_dict = {self.x_in: x_tt, self.y_: y})
                    self.writer.add_summary(summary, i)

                except:
                    print("checkpoint didnt saved. pleas be sure that is well saved")

        self.writer.close()

        return 0



    def accuracy_score(self, x_test, y_test):
        channel_index = (4, 5, 9, 17, 57)
        x_test = x_test[:,channel_index,:,:]
        x_tt = np.zeros((x_test.shape[0], 61*5, 61, 1))
        for i in range(x_test.shape[3]):
            x_tt [:,:,i,:] = x_test[:,:,:,i].reshape((-1,x_test.shape[2]*x_test.shape[1], 1))

        # self.out_softmax = tf.nn.softmax()
        y_mod = np.zeros((y_test.shape[0],2), dtype=np.int16)
        y_test = y_test.astype(np.float32)
        for i in range(y_test.shape[0]):
            y_mod[i, int(((y_test[i]) + 1)/2)] = 1
        # y = tf.Variable(y_mod,dtype=tf.float32)
        y__ = y_mod.astype(np.float32)
        y_pred = np.argmax(self.sess.run(self.y, feed_dict={self.x_in: x_tt}),axis=1)
        y_pred2 = np.argmax(self.sess.run(tf.nn.softmax(self.y), feed_dict={self.x_in: x_tt}),axis=1)
        self.acc = np.average(np.equal(y_pred, np.argmax(y__, axis = 1) ))

        return self.acc
"""


class CNN1D:
    # origianl input data size : 61x61x64(frequency x time point x channel number) to 1D image
    # reference : A novel deep learning approach for classification of EEG motor imagerty signals, Yousef Razael Tabar and Ugur Halici, 2017, Journal of Neural Engineering
    #
    def __init__(self, meta_json, save_dir, sub=0, cv=0, gap=False):
        input_layer = 3
        self.meta_json = meta_json
        self.ccc = 0
        self.save_dir = save_dir
        self.log_sub = sub
        self.log_cv = cv
        self.sub = sub
        self.cv = cv
        self.concat = gap

        return

    def initialize_variables(self):
        # reshape x_train to 2D image
        # For this case, use DNN with 5 channels only : E6(Fz), E10(Fp1), E5(Fp2), E18(F7), E58(F8)
        # x_train = np.zeros((100,64,61,61))
        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

        channel_index = (4, 5, 9, 17, 57)

        # conv network
        self.x_in = tf.placeholder(shape = [None, 61*5, 61, 1], dtype=tf.float32)
        self.y_ = tf.placeholder(dtype=tf.float32, shape = [None, 2])

        w1 = tf.Variable(tf.truncated_normal(shape = (61*5,7,1,60) , stddev=0.1))
        bias1 = tf.Variable(tf.constant(0.1, shape = [60]))
        conv1 = tf.nn.conv2d(self.x_in, w1, strides = [1,1,1,1], padding="VALID")
        conv1_out = tf.nn.relu(conv1 + bias1)
        pool1 = tf.nn.max_pool(conv1_out, ksize= [1,1,5,1], strides=[1,1,5,1], padding='SAME')

        # This model cannot be structurally deep since this model applied 1d convolution
        self.weight_ = tf.Variable(tf.zeros((660,2),dtype=tf.float32),dtype=tf.float32)
        self.bias_ = tf.Variable(tf.zeros((2),dtype=tf.float32),dtype=tf.float32)
        self.pool1_flat = tf.reshape(pool1, shape=[-1, 660])
        self.y = tf.add(tf.matmul(self.pool1_flat, self.weight_), self.bias_)

        # cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.y, labels=self.y_))

        self.train_step = tf.train.AdamOptimizer(learning_rate=self.meta_json["LEARNING_RATE"]).minimize(self.loss)

        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()

        return


    def restoring_variables(self):
        som = [f for f in os.listdir(self.meta_json["MODEL_NAME"] + '-'  + self.meta_json["STARTED_DATESTRING"] + '/.') if f.lower().endswith('iter.ckpt')]
        soi = []
        for i in som:
            soi.append(int(i.split('.')[0][5:-4]))
        self.saver.restore(self.sess, self.meta_json["MODEL_NAME"] + '-'  + self.meta_json["STARTED_DATESTRING"] + "/model.ckpt")
        self.ccc = 100

        return


    def train(self, x_train, y_train, x_test, y_test):

        if self.meta_json["rest"] is True:
            self.initialize_variables()
            self.restoring_variables()
        else:
            self.initialize_variables()

        channel_index = (4, 5, 9, 17, 57)
        x_train_ = x_train[:,channel_index,:,:]
        x_tt = np.zeros((x_train_.shape[0], 61*5, 61, 1))
        for i in range(x_train_.shape[3]):
            x_tt [:,:,i,:] = x_train_[:,:,:,i].reshape((-1,x_train_.shape[2]*x_train_.shape[1], 1))

        y_mod = np.zeros((y_train.shape[0],2), dtype=np.int16)
        y_train = y_train.astype(np.float32)
        for i in range(y_train.shape[0]):
            y_mod[i, int(((y_train[i]) + 1)/2)] = 1
        # y = tf.Variable(y_mod,dtype=tf.float32)
        y = y_mod.astype(np.float32)


        loss_ = self.loss
        loss_hist = tf.summary.scalar('loss', loss_)
        merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(
            './' + self.meta_json["MODEL_NAME"] + self.meta_json["STARTED_DATESTRING"] + self.meta_json["LOGDIR_ROOT"],
            self.sess.graph)

        # x_tt = x_train.reshape((x_train.shape[0], -1) ).astype(np.float32)
        # x_tt = tf.Variable( tf.cast( tf.reshape(x_train, (x_train.shape[0], -1) ), dtype=tf.float32) )
        y_mod = np.zeros((y_train.shape[0],2), dtype=np.int16)
        y_train = y_train.astype(np.float32)
        for i in range(y_train.shape[0]):
            y_mod[i, int(((y_train[i]) + 1)/2)] = 1
        # y = tf.Variable(y_mod,dtype=tf.float32)
        y = y_mod.astype(np.float32)
        for i in range(int(self.meta_json["NUM_STEPS"])):
            self.sess.run(self.train_step, feed_dict = {self.x_in: x_tt, self.y_: y})
            if i%(1000) == 0:
                self.ccc += 1
                print('[{0}/{1}] : {2} -- loss : {3}'.format(self.ccc , int(self.meta_json["NUM_STEPS"]/1000) , self.accuracy_score( x_test, y_test), self.sess.run(self.loss, feed_dict = {self.x_in: x_tt, self.y_: y}) ) )
                try:
                    save_path = self.saver.save(self.sess, self.meta_json["MODEL_NAME"] + '-' + self.meta_json[
                        "STARTED_DATESTRING"] + "/model.ckpt")
                    save_path = self.saver.save(self.sess, self.meta_json["MODEL_NAME"] + '-' + self.meta_json[
                        "STARTED_DATESTRING"] + "/model" + str(i) + "iter.ckpt")
                    summary = self.sess.run(merged, feed_dict = {self.x_in: x_tt, self.y_: y})
                    self.writer.add_summary(summary, i)

                except:
                    print("checkpoint didnt saved. pleas be sure that is well saved")

        self.writer.close()

        return 0



    def accuracy_score(self, x_test, y_test):
        channel_index = (4, 5, 9, 17, 57)
        x_test = x_test[:,channel_index,:,:]
        x_tt = np.zeros((x_test.shape[0], 61*5, 61, 1))
        for i in range(x_test.shape[3]):
            x_tt [:,:,i,:] = x_test[:,:,:,i].reshape((-1,x_test.shape[2]*x_test.shape[1], 1))

        # self.out_softmax = tf.nn.softmax()
        y_mod = np.zeros((y_test.shape[0],2), dtype=np.int16)
        y_test = y_test.astype(np.float32)
        for i in range(y_test.shape[0]):
            y_mod[i, int(((y_test[i]) + 1)/2)] = 1
        # y = tf.Variable(y_mod,dtype=tf.float32)
        y__ = y_mod.astype(np.float32)
        y_pred = np.argmax(self.sess.run(self.y, feed_dict={self.x_in: x_tt}),axis=1)
        y_pred2 = np.argmax(self.sess.run(tf.nn.softmax(self.y), feed_dict={self.x_in: x_tt}),axis=1)
        self.acc = np.average(np.equal(y_pred, np.argmax(y__, axis = 1) ))

        return self.acc


class CNN2D(CNN1D):
    # 61x61x64(frequency x time point x channel number) to 2D image
    def initialize_variables(self):
        # reshape x_train to 2D image
        # For DNN with 5 channels only : E6(Fz), E10(Fp1), E5(Fp2), E18(F7), E58(F8)
        # For CNN2D with 4 channels only : E10(Fp1), E5(Fp2), E18(F7), E58(F8)
        # x_train = np.zeros((100,64,61,61))
        # shape of the image = (batch, 61*2, 61*2)

        # # controls the learning rate decay.
        # batch = tf.Variable(0)
        #
        # learning_rate = tf.train.exponential_decay(
        #     1e-4,  # Base learning rate.
        #     batch * batch_size,  # Current index into the dataset.
        #     train_size,  # Decay step.
        #     0.95,  # Decay rate.
        #     staircase=True)
        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

        channel_index = (4, 9, 17, 57)

        # conv network




        # conv network
        self.x_in = tf.placeholder(shape = [None, 61*2, 61*2, 1], dtype=tf.float32)
        self.y_ = tf.placeholder(dtype=tf.float32, shape = [None, 2])


        w1 = tf.Variable(tf.truncated_normal(shape = (7,7,1,16) , stddev=0.1))
        bias1 = tf.Variable(tf.constant(0.1, shape = [16]))
        conv1 = tf.nn.conv2d(self.x_in, w1, strides = [1,1,1,1], padding="VALID") # (?, 116, 116, 16)
        conv1_out = tf.nn.relu(conv1 + bias1) # (?, 116, 116, 16)
        pool1 = tf.nn.max_pool(conv1_out, ksize= [1,3,3,1], strides=[1,2,2,1], padding='SAME')# (?, 58, 58, 16)

        w2 = tf.Variable(tf.truncated_normal(shape = (7,7,16,32) , stddev=0.1))
        bias2 = tf.Variable(tf.constant(0.1, shape = [32]))
        conv2 = tf.nn.conv2d(pool1, w2, strides = [1,1,1,1], padding="VALID") # (?, 52, 52, 32)
        conv2_out = tf.nn.relu(conv2 + bias2) # (?, 52, 52, 32)
        pool2 = tf.nn.max_pool(conv2_out, ksize= [1,3,3,1], strides=[1,2,2,1], padding='SAME')# (?, 26, 26, 32)

        w3 = tf.Variable(tf.truncated_normal(shape = (5,5,32,48) , stddev=0.1))
        bias3 = tf.Variable(tf.constant(0.1, shape = [48]))
        conv3 = tf.nn.conv2d(pool2, w3, strides = [1,1,1,1], padding="VALID") # (?, 22, 22, 48)
        conv3_out = tf.nn.relu(conv3 + bias3) # (?, 22, 22, 48)
        pool3 = tf.nn.max_pool(conv3_out, ksize= [1,3,3,1], strides=[1,2,2,1], padding='SAME')# (?, 11, 11, 48)

        w4 = tf.Variable(tf.truncated_normal(shape = (5,5,48,64) , stddev=0.1))
        bias4 = tf.Variable(tf.constant(0.1, shape = [64]))
        conv4 = tf.nn.conv2d(pool3, w4, strides = [1,1,1,1], padding="VALID") # (?, 7, 7, 64)
        conv4_out = tf.nn.relu(conv4 + bias4) # (?, 7, 7, 64)
        pool4 = tf.nn.max_pool(conv4_out, ksize= [1,3,3,1], strides=[1,2,2,1], padding='SAME')# (?, 4, 4, 64)


        self.out_layer = tf.reshape(pool4, shape = (-1,1024))
        # This model cannot be structurally deep since this model applied 1d convolution
        self.weight_ = tf.Variable(tf.zeros((1024,2),dtype=tf.float32),dtype=tf.float32)
        self.bias_ = tf.Variable(tf.zeros((2),dtype=tf.float32),dtype=tf.float32)


        self.y = tf.add(tf.matmul(self.out_layer, self.weight_), self.bias_)

        # cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.y, labels=self.y_))

        self.train_step = tf.train.AdamOptimizer(learning_rate=self.meta_json["LEARNING_RATE"]).minimize(self.loss)

        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()

        return

    def restoring_variables(self):
        som = [f for f in os.listdir(self.meta_json["MODEL_NAME"] + '-'  + self.meta_json["STARTED_DATESTRING"] + '/.') if f.lower().endswith('iter.ckpt')]
        soi = []
        for i in som:
            soi.append(int(i.split('.')[0][5:-4]))
        self.saver.restore(self.sess, self.meta_json["MODEL_NAME"] + '-'  + self.meta_json["STARTED_DATESTRING"] + "/model.ckpt")
        self.ccc = 100

        return


    def train(self, x_train, y_train, x_test, y_test):

        if self.meta_json["rest"] is True:
            self.initialize_variables()
            self.restoring_variables()
        else:
            self.initialize_variables()

        channel_index = (4, 9, 17, 57)
        temp1 =np.concatenate((x_train[:,4,:,:],x_train[:,9,:,:]), axis=1)
        temp2 =np.concatenate((x_train[:,17,:,:],x_train[:,57,:,:]), axis=1)
        x_tt = np.concatenate((temp1,temp2),axis=2).reshape((x_train.shape[0],121*2,121*2,1))

        y_mod = np.zeros((y_train.shape[0],2), dtype=np.int16)
        y_train = y_train.astype(np.float32)
        for i in range(y_train.shape[0]):
            y_mod[i, int(((y_train[i]) + 1)/2)] = 1
        # y = tf.Variable(y_mod,dtype=tf.float32)
        y = y_mod.astype(np.float32)

        loss_ = self.loss
        loss_hist = tf.summary.scalar('loss', loss_)
        merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(
            './' + self.meta_json["MODEL_NAME"] + self.meta_json["STARTED_DATESTRING"] + self.meta_json["LOGDIR_ROOT"],
            self.sess.graph)

        # x_tt = tf.Variable( tf.cast( tf.reshape(x_train, (x_train.shape[0], -1) ), dtype=tf.float32) )
        y_mod = np.zeros((y_train.shape[0],2), dtype=np.int16)
        y_train = y_train.astype(np.float32)
        for i in range(y_train.shape[0]):
            y_mod[i, int(((y_train[i]) + 1)/2)] = 1
        # y = tf.Variable(y_mod,dtype=tf.float32)
        y = y_mod.astype(np.float32)
        for i in range(int(self.meta_json["NUM_STEPS"])):
            self.sess.run(self.train_step, feed_dict = {self.x_in: x_tt, self.y_: y})
            if i%(1000) == 0:
                self.ccc += 1
                print('[{0}/{1}] : {2} -- loss : {3}'.format(self.ccc , int(self.meta_json["NUM_STEPS"]/1000) , self.accuracy_score(x_test, y_test), self.sess.run(self.loss, feed_dict = {self.x_in: x_tt, self.y_: y}) ) )
                try:
                    save_path = self.saver.save(self.sess, self.meta_json["MODEL_NAME"] + '-' + self.meta_json[
                        "STARTED_DATESTRING"] + "/model.ckpt")
                    save_path = self.saver.save(self.sess, self.meta_json["MODEL_NAME"] + '-' + self.meta_json[
                        "STARTED_DATESTRING"] + "/model" + str(i) + "iter.ckpt")
                    summary = self.sess.run(merged, feed_dict={self.x_in: x_tt, self.y_: y})
                    self.writer.add_summary(summary, i)
                except:
                    print("checkpoint didnt saved. pleas be sure that is well saved")

        self.writer.close()

        return 0

    def accuracy_score(self, x_test, y_test):
        channel_index = (4, 9, 17, 57)
        temp1 =np.concatenate((x_test[:,4,:,:],x_test[:,9,:,:]), axis=1)
        temp2 =np.concatenate((x_test[:,17,:,:],x_test[:,57,:,:]), axis=1)
        x_tt = np.concatenate((temp1,temp2),axis=2).reshape((x_test.shape[0],121*2*2,121*2*2,1)) #original ...,121*2,121*2,1))
        # x_tt = np.concatenate((temp1,temp2),axis=2)
        # x_tt = np.kron(x_tt, np.ones((2,2))).reshape((x_test.shape[0],121*2*2,121*2*2,1)) #original ...,121*2,121*2,1))

        y_mod = np.zeros((y_test.shape[0],2), dtype=np.int16)
        y_test = y_test.astype(np.float32)
        for i in range(y_test.shape[0]):
            y_mod[i, int(((y_test[i]) + 1)/2)] = 1
        # y = tf.Variable(y_mod,dtype=tf.float32)
        y = y_mod.astype(np.float32)


        channel_index = (4, 9, 17, 57)
        temp1 =np.concatenate((x_test[:,4,:,:],x_test[:,9,:,:]), axis=1)
        temp2 =np.concatenate((x_test[:,17,:,:],x_test[:,57,:,:]), axis=1)
        x_tt = np.concatenate((temp1,temp2),axis=2).reshape((x_test.shape[0],121*2*2,121*2*2,1))

        # y = tf.Variable(y_mod,dtype=tf.float32)
        y__ = y_mod.astype(np.float32)
        y_pred = np.argmax(self.sess.run(self.y, feed_dict={self.x_in: x_tt}),axis=1)
        self.acc = np.average(np.equal(y_pred, np.argmax(y__, axis = 1) ))

        return self.acc


class CNN2DwithCAM(CNN2D):
    def restoring_variables(self):
        # som = [f for f in os.listdir(self.meta_json["MODEL_NAME"] + '-'  + self.meta_json["STARTED_DATESTRING"] + '/acc_sub' + str(self.sub)) if f.lower().endswith('iter.ckpt')]
        som = []
        for f in os.listdir(self.meta_json["MODEL_NAME"] + '-'  + self.meta_json["STARTED_DATESTRING"] + '/acc_sub' + str(self.sub)):
            if fnmatch.fnmatch(f, '*iter.ckpt.meta'):
                som.append(f)
        self.saver.restore(self.sess, self.meta_json["MODEL_NAME"] + '-'  + self.meta_json["STARTED_DATESTRING"]+ '/acc_sub' + str(self.sub) + '/' + som[-1][:-5])

    def soms(self):
        som = []
        somename=[]
        for f in os.listdir('./cnn2d_soms' + '/sub' + str(self.sub)):
            if fnmatch.fnmatch(f, '*iter.ckpt.meta'):
                som.append('./cnn2d_soms' + '/sub' + str(self.sub) + '/' + f)
                somename.append(f[5:-14])
        return som, somename

    def initialize_variables(self):
        tf.reset_default_graph()

        channel_index = (4, 9, 17, 57)
        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

        # conv network
        self.x_in = tf.placeholder(shape=[None, 121 * 2, 121 * 2, 1], dtype=tf.float32)  # original x2
        self.y_ = tf.placeholder(dtype=tf.float32, shape=[None, 2])
        if self.concat:
            self.gap_ = tf.placeholder(dtype=tf.float32, shape=[None, 30])

        w1 = tf.Variable(tf.truncated_normal(shape=(7, 7, 1, 10), stddev=0.1))
        bias1 = tf.Variable(tf.constant(0.1, shape=[10]))

        conv1 = tf.nn.conv2d(self.x_in, w1, strides=[1, 1, 1, 1], padding="VALID")  # (?, 471, 471, 10)
        conv1_out = tf.nn.relu(conv1)
        pool1 = tf.nn.max_pool(conv1_out, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')  # (?, 236, 236, 10)
        w2 = tf.Variable(tf.truncated_normal(shape=(7, 7, 10, 30), stddev=0.1))
        bias2 = tf.Variable(tf.constant(0.1, shape=[30]))

        conv2 = tf.nn.conv2d(pool1, w2, strides=[1, 1, 1, 1], padding="VALID")  # (?, 230, 230, 30)
        conv2_out = tf.nn.relu(conv2)
        self.pool2 = tf.nn.max_pool(conv2_out, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                                    padding='SAME')  # (?, 115, 115, 30)

        # This model cannot be structurally deep since this model applied 1d convolution
        self.gap__ = tf.reduce_mean(tf.reshape(self.pool2, shape=(-1, 115 * 115, 30)), axis=1)

        if self.concat:
            self.weight_ = tf.Variable(tf.zeros((60, 2), dtype=tf.float32), dtype=tf.float32)
            self.gap = tf.concat([self.gap__, self.gap_], 1)
        else:
            self.weight_ = tf.Variable(tf.zeros((30,2), dtype=tf.float32), dtype=tf.float32)
            self.gap = self.gap__
        self.bias_ = tf.Variable(tf.zeros((2), dtype=tf.float32), dtype=tf.float32)
        self.y = tf.matmul(self.gap, self.weight_)

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.y, labels=self.y_))
        self.train_step = tf.train.AdamOptimizer(learning_rate=self.meta_json["LEARNING_RATE"]).minimize(self.loss)
        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()

        return

    def CAMgen(self, x_test, y_test):
        # none 12 12 30
        # num_feature = self.weight_.shape[0]
        # num_classes = self.weight_.shape[1]
        temp1 =np.concatenate((x_test[:,4,:,:],x_test[:,9,:,:]), axis=1)
        temp2 =np.concatenate((x_test[:,17,:,:],x_test[:,57,:,:]), axis=1)
        x_tt = np.concatenate((temp1,temp2),axis=2).reshape((x_test.shape[0],121*2,121*2,1)) # original 121*2

        y_mod = np.zeros((y_test.shape[0],2), dtype=np.int16)
        y_test = y_test.astype(np.float32)
        for i in range(y_test.shape[0]):
            y_mod[i, int(((y_test[i]) + 1)/2)] = 1
        # y = tf.Variable(y_mod,dtype=tf.float32)
        y = y_mod.astype(np.float32)

        weights_ = self.sess.run(self.weight_)
        pool2seed = self.sess.run(self.pool2, feed_dict={self.x_in: x_tt})  # batch,17,17,30
        self.image = np.zeros((115, 115, 2))
        for bi in range(pool2seed.shape[0]):
            for i in range(self.weight_.shape[0]):
                self.image[:, :, np.where(y[bi] == 1)[0].tolist()[0]] += (pool2seed[bi,:,:,:].reshape((115,115,30))[:,:,i] * weights_ [i,np.where(y[bi] == 1)[0].tolist()[0]])

        # devide im 1000
        self.image = self.image - np.min(self.image)
        denom = np.max(self.image)/1000
        self.image = np.int16(self.image/denom)

        # self.save_dir = './PJS'
        np.save(self.save_dir + '/CAM_{0}'.format(self.ccc) ,self.image)

        return 0

    def accuracy_score(self, x_test, y_test, g_test=[]):
        # channel_index = (0, 1, 2, 3) = (Fp1, Fp2, F7, F8)
        temp1 =np.concatenate((x_test[:,0,:,:],x_test[:,1,:,:]), axis=1)
        temp2 =np.concatenate((x_test[:,2,:,:],x_test[:,3,:,:]), axis=1)
        x_tt = np.concatenate((temp1,temp2),axis=2).reshape((x_test.shape[0],121*2,121*2,1)) #original ...,121*2,121*2,1))
        # x_tt = np.concatenate((temp1,temp2),axis=2)
        # x_tt = np.kron(x_tt, np.ones((2,2))).reshape((x_test.shape[0],121*2*2,121*2*2,1)) #original ...,121*2,121*2,1))

        y_mod = np.zeros((y_test.shape[0],2), dtype=np.int16)
        y_test = y_test.astype(np.float32)
        for i in range(y_test.shape[0]):
            y_mod[i, int(((y_test[i]) + 1)/2)] = 1
        # y = tf.Variable(y_mod,dtype=tf.float32)
        y = y_mod.astype(np.float32)

        # y = tf.Variable(y_mod,dtype=tf.float32)
        y__ = y_mod.astype(np.float32)
        # y_pred = np.argmax(self.sess.run(self.y, feed_dict={self.x_in: x_tt}),axis=1)

        totbat = x_tt.shape[0]
        fortemp = 1.0
        while (totbat / fortemp) > 200:
            fortemp += 1
        if fortemp == 1.0:
            e_bat = totbat
        else:
            e_bat = int(totbat / fortemp) +1
        # e_batt = int(x_ttest.shape[0] / fortemp)
        y_pred_temp = []
        y_pred_temp = np.array([])

        testloss = 0.0
        if self.concat:
            for j in range(int(fortemp)):
                temp = np.argmax(
                    self.sess.run(self.y, feed_dict={self.x_in: x_tt[e_bat * j:e_bat * (j + 1), :, :, :],
                                                     self.gap_: g_test[e_bat * j:e_bat * (j + 1), :]}), axis=1)
                if j == 0:
                    y_pred_temp = temp
                else:
                    y_pred_temp = np.concatenate((y_pred_temp, temp), axis=0)
        else:
            for j in range(int(fortemp)):
                temp = np.argmax(
                    self.sess.run(self.y, feed_dict={self.x_in: x_tt[e_bat * j:e_bat * (j + 1), :, :, :]}), axis=1)
                if j == 0:
                    y_pred_temp = temp
                else:
                    y_pred_temp = np.concatenate((y_pred_temp, temp), axis=0)
        if self.concat:
            for j in range(int(fortemp)):
                testloss += self.sess.run(self.loss,
                                          feed_dict={self.x_in: x_tt[e_bat * j:e_bat * (j + 1), :, :, :],
                                                     self.y_: y[e_bat * j:e_bat * (j + 1), :],
                                                     self.gap_: g_test[e_bat * j:e_bat * (j + 1), :]})
        else:
            for j in range(int(fortemp)):
                testloss += self.sess.run(self.loss,
                                          feed_dict={self.x_in: x_tt[e_bat * j:e_bat * (j + 1), :, :, :],
                                                     self.y_: y[e_bat * j:e_bat * (j + 1), :]})
        y_pred = y_pred_temp
        shp = y_pred.shape[0]
        self.testloss = testloss
        self.acc = np.average(np.equal(y_pred, np.argmax(y__[:shp,:], axis = 1) ))

        return self.acc

    def train(self, x_train, y_train, x_test, y_test, g_train=[], g_test=[]):
        # x_train
        # channel_index = (0, 1, 2, 3) = (Fp1, Fp2, F7, F8)
        temp1 = np.concatenate((x_train[:, 0, :, :], x_train[:, 1, :, :]), axis=1)
        temp2 = np.concatenate((x_train[:, 2, :, :], x_train[:, 3, :, :]), axis=1)
        x_tt = np.concatenate((temp1, temp2), axis=2).reshape(
            (x_train.shape[0], 121 * 2, 121 * 2, 1))  # original ...,121*2,121*2,1))

        # y_train
        y_mod = np.zeros((y_train.shape[0], 2), dtype=np.int16)
        y_train = y_train.astype(np.float32)
        for i in range(y_train.shape[0]):
            y_mod[i, int(((y_train[i]) + 1) / 2)] = 1
        # y = tf.Variable(y_mod,dtype=tf.float32)
        y = y_mod.astype(np.float32)

        loss_ = self.loss
        loss_hist = tf.summary.scalar('loss', loss_)
        merged = tf.summary.merge_all()

        self.writer = tf.summary.FileWriter(
            './' + self.meta_json["MODEL_NAME"] + self.meta_json["STARTED_DATESTRING"] + self.meta_json[
                "LOGDIR_ROOT"] + '/dataset{0}'.format(self.log_sub) + '/cv{0}'.format(self.log_cv),
            self.sess.graph)
        self.writer.add_graph(self.sess.graph)

        for i in range(int(self.meta_json["NUM_STEPS"])):
            totbat = x_tt.shape[0]
            fortemp = 1.0
            while (totbat / fortemp) > 100:
                fortemp += 1
            if fortemp == 1.0:
                e_bat = totbat
            else:
                e_bat = int(totbat / fortemp) +1
            # e_batt = int(x_ttest.shape[0] / fortemp)
            if self.concat:
                for j in range(int(fortemp)):
                    self.sess.run(self.train_step, feed_dict={self.x_in: x_tt[e_bat * j:e_bat * (j + 1), :, :, :],
                                                              self.y_: y[e_bat * j:e_bat * (j + 1), :],
                                                              self.gap_: g_train[e_bat * j:e_bat * (j + 1), :]})
            else:
                for j in range(int(fortemp)):
                    self.sess.run(self.train_step, feed_dict={self.x_in: x_tt[e_bat * j:e_bat * (j + 1), :, :, :],
                                                              self.y_: y[e_bat * j:e_bat * (j + 1), :]})
            temploss = 0.0
            if self.concat:
                for k in range(int(fortemp)):
                    temploss += self.sess.run(self.loss,
                                              feed_dict={self.x_in: x_tt[e_bat * k:e_bat * (k + 1), :, :, :],
                                                         self.y_: y[e_bat * k:e_bat * (k + 1), :],
                                                         self.gap_: g_train[e_bat * k:e_bat * (k + 1), :]})
            else:
                for k in range(int(fortemp)):
                    temploss += self.sess.run(self.loss,
                                              feed_dict={self.x_in: x_tt[e_bat * k:e_bat * (k + 1), :, :, :],
                                                         self.y_: y[e_bat * k:e_bat * (k + 1), :]})
            if self.concat:
                testacc = self.accuracy_score(x_test, y_test, g_test)
            else:
                testacc = self.accuracy_score(x_test, y_test)
            if i == 0 or (i+1) % (self.meta_json["CHECKPOINT_EVERY"]) == 0:
                print('[{0}/{1}] : test_acc : {2} training_loss : {3} test_loss : {4}'.format(self.ccc, int(
                    self.meta_json["NUM_STEPS"] / self.meta_json["CHECKPOINT_EVERY"]), testacc, temploss, self.testloss))
                self.ccc += 1
                try:
                    save_path = self.saver.save(self.sess, self.save_dir + '/dataset{0}'.format(self.sub) + "/model.ckpt")
                    save_path = self.saver.save(self.sess, self.save_dir + '/dataset{0}'.format(self.sub) + "/model" + str(i) + "iter.ckpt")
                    print('train completion\n')
                except Exception as e:
                    print(e)
                    print("checkpoint didnt saved. pleas be sure that is well saved")

            summarytrl = tf.Summary(value=[tf.Summary.Value(tag="training_batch_loss", simple_value=temploss)])
            summarytel = tf.Summary(value=[tf.Summary.Value(tag="test_loss", simple_value=self.testloss)])
            # summarytra = tf.Summary(value=[tf.Summary.Value(tag="training_batch_acc", simple_value=trainacc)])
            summarytea = tf.Summary(value=[tf.Summary.Value(tag="test_acc", simple_value=testacc)])
            self.writer.add_summary(summarytrl, i)
            self.writer.add_summary(summarytel, i)
            # self.writer.add_summary(summarytra, i)
            self.writer.add_summary(summarytea, i)

        self.writer.close()

        return 0

    def get_gap(self, X, Y):
        if self.concat:
            raise ValueError

        # x_train
        # channel_index = (0, 1, 2, 3) = (Fp1, Fp2, F7, F8)
        temp1 = np.concatenate((X[:, 0, :, :], X[:, 1, :, :]), axis=1)
        temp2 = np.concatenate((X[:, 2, :, :], X[:, 3, :, :]), axis=1)
        x_tt = np.concatenate((temp1, temp2), axis=2).reshape(
            (X.shape[0], 121 * 2, 121 * 2, 1))  # original ...,121*2,121*2,1))

        # y_train
        y_mod = np.zeros((Y.shape[0], 2), dtype=np.int16)
        Y = Y.astype(np.float32)
        for i in range(Y.shape[0]):
            y_mod[i, int(((Y[i]) + 1) / 2)] = 1
        y = y_mod.astype(np.float32)

        totbat = x_tt.shape[0]
        fortemp = 1.0
        while (totbat / fortemp) > 100:
            fortemp += 1
        if fortemp == 1.0:
            e_bat = totbat
        else:
            e_bat = int(totbat / fortemp) + 1

        gap = []
        for j in range(int(fortemp)):
            temp = self.sess.run(self.gap, feed_dict={self.x_in: x_tt[e_bat * j:e_bat * (j + 1), :, :, :],
                                                      self.y_: y[e_bat * j:e_bat * (j + 1), :]})
            if j == 0:
                gap = temp
            else:
                gap = np.concatenate((gap, temp), axis=0)

        return gap

    def set_gap(self, gap):
        if not self.concat:
            raise ValueError
        self.w_gap = gap

"""
class CNN3D_TFC :
    # not the freq as 61, 30
    def __init__(self, meta_json, save_dir, sub=0, cv=0):
        input_layer = 3
        self.meta_json = meta_json
        self.ccc = 0
        self.save_dir = save_dir
        self.sub = sub
        self.cv = cv

        return

    def restoring_variables(self):
        som = [f for f in os.listdir(self.meta_json["MODEL_NAME"] + '-'  + self.meta_json["STARTED_DATESTRING"] + '/.') if f.lower().endswith('iter.ckpt')]
        soi = []
        for i in som:
            soi.append(int(i.split('.')[0][5:-4]))
        self.saver.restore(self.sess, self.meta_json["MODEL_NAME"] + '-'  + self.meta_json["STARTED_DATESTRING"] + "/model.ckpt")
        self.ccc = 100

        return

    def initialize_variables(self):
        # reshape x_train to 2D image
        # For DNN with 5 channels only : E6(Fz), E10(Fp1), E5(Fp2), E18(F7), E58(F8)
        # For CNN2D with 4 channels only : E10(Fp1), E5(Fp2), E18(F7), E58(F8)
        # x_train = np.zeros((100,64,61,61))
        # shape of the image = (batch, 61*2, 61*2)

        # # controls the learning rate decay.
        # batch = tf.Variable(0)
        #
        # learning_rate = tf.train.exponential_decay(
        #     1e-4,  # Base learning rate.
        #     batch * batch_size,  # Current index into the dataset.
        #     train_size,  # Decay step.
        #     0.95,  # Decay rate.
        #     staircase=True)
        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

        channel_index = (4, 9, 17, 57)

        # conv network


        # conv network
        self.x_in = tf.placeholder(shape = [None, 30, 30, 30, 1], dtype=tf.float32)
        self.y_ = tf.placeholder(dtype=tf.float32, shape = [None, 2])

        # with tf.device('/cpu:0'):
        w1 = tf.Variable(tf.truncated_normal(shape = (7,7,7,1,4) , stddev=0.1))
        bias1 = tf.Variable(tf.constant(0.1, shape = [4]))
        conv1 = tf.nn.conv3d(self.x_in, w1, strides = [1,1,1,1,1], padding="VALID") # (?, 116, 116, 16)
        conv1_out = tf.nn.relu(conv1 + bias1) # (?, 116, 116, 16)
        pool1 = tf.nn.max_pool3d(conv1_out, ksize= [1,3,3,3,1], strides=[1,1,1,1,1], padding='SAME')# (?, 7,6,6,20)

        w11 = tf.Variable(tf.truncated_normal(shape = (7,7,7,4,8) , stddev=0.1))
        bias11 = tf.Variable(tf.constant(0.1, shape = [8]))
        conv11 = tf.nn.conv3d(pool1, w11, strides = [1,1,1,1,1], padding="VALID") # (?, 116, 116, 16)
        conv11_out = tf.nn.relu(conv11 + bias11) # (?, 116, 116, 16)
        pool2 = tf.nn.max_pool3d(conv11_out, ksize= [1,3,3,3,1], strides=[1,2,2,2,1], padding='SAME')# (?, 7,6,6,20)

        self.out_layer = tf.reshape(pool2, shape = (-1,5832))
        # This model cannot be structurally deep since this model applied 1d convolution
        self.weight_ = tf.Variable(tf.zeros((5832,2),dtype=tf.float32),dtype=tf.float32)
        self.bias_ = tf.Variable(tf.zeros((2),dtype=tf.float32),dtype=tf.float32)


        self.y = tf.add(tf.matmul(self.out_layer, self.weight_), self.bias_)

        # cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.y, labels=self.y_))

        self.train_step = tf.train.AdamOptimizer(learning_rate=self.meta_json["LEARNING_RATE"]).minimize(self.loss)

        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()

        return

    def train(self, x_train, y_train, x_test, y_test):
        ch = np.array([63,62,64,17,10,5,1,61,8,11,2,18,13,9,6,3,59,58,12,60,19,14,57,56,23,15,7,54,53,55])
        channel_index = tuple((ch-1).tolist())
        if self.meta_json["rest"] is True:
            self.initialize_variables()
            self.restoring_variables()
        else:
            self.initialize_variables()
        x_train = x_train.reshape((x_train.shape[0],x_train.shape[1],x_train.shape[2],x_train.shape[3],1))
        x_trains = x_train[:, channel_index, :, :30, :]
        x_train = np.zeros((x_train.shape[0], 30, 30, x_trains.shape[3], 1)) # ? 64 30 30 1
        for tr in range(x_train.shape[2]):
            x_train[:,:,tr,:,:] = (x_trains[:,:,2*tr,:,:] +x_trains[:,:,2*tr+1,:,:]) /2
        # x_tr = np.zeros(shape=(x_train.shape[0], x_train.shape[2], x_train.shape[3], x_train.shape[1],1))
        # for chi in range(x_train.shape[1]):
        #     x_tr[:,:,:,chi,0] = x_train[:,chi,:,:]
        # x_train = x_tr
        # del x_tr

        y_mod = np.zeros((y_train.shape[0],2), dtype=np.int16)
        y_train = y_train.astype(np.float32)
        for i in range(y_train.shape[0]):
            y_mod[i, int(((y_train[i]) + 1)/2)] = 1
        # y = tf.Variable(y_mod,dtype=tf.float32)
        y = y_mod.astype(np.float32)

        loss_ = self.loss
        loss_hist = tf.summary.scalar('loss', loss_)
        merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(
            './' + self.meta_json["MODEL_NAME"] + self.meta_json["STARTED_DATESTRING"] + self.meta_json["LOGDIR_ROOT"],
            self.sess.graph)

        # x_tt = tf.Variable( tf.cast( tf.reshape(x_train, (x_train.shape[0], -1) ), dtype=tf.float32) )
        # y_mod = np.zeros((y_train.shape[0],2), dtype=np.int16)
        # y_train = y_train.astype(np.float32)
        # for i in range(y_train.shape[0]):
        #     y_mod[i, int(((y_train[i]) + 1)/2)] = 1
        # # y = tf.Variable(y_mod,dtype=tf.float32)
        # y = y_mod.astype(np.float32)
        for i in range(int(self.meta_json["NUM_STEPS"])):

            totbat = x_train.shape[0]
            fortemp = 1.0
            while (totbat / fortemp) > 10:
                fortemp += 1
            e_bat = int(totbat / fortemp) + 1


            for j in range(int(fortemp)):
                self.sess.run(self.train_step, feed_dict = {self.x_in: x_train[e_bat*j:e_bat*(j+1),:,:,:,:], self.y_: y[e_bat*j:e_bat*(j+1),:]})
            if i%(1000) == 0:
                self.ccc += 1
                totbat_ = x_test.shape[0]
                fortemp_ = 1.0
                while (totbat_ / fortemp_) > 20:
                    fortemp_ += 1
                e_bat_ = int(totbat_ / fortemp_)
                dap = []
                for k in range(int(fortemp_)):
                    dap.append(self.accuracy_score(x_test[e_bat*k:e_bat*(k+1),:,:,:], y_test[e_bat*k:e_bat*(k+1)]))
                ls =  0
                for j in range(int(fortemp)):
                    ls += self.sess.run(self.loss, feed_dict={self.x_in: x_train[e_bat*j:e_bat*(j+1),:,:,:,:], self.y_: y[e_bat*j:e_bat*(j+1),:]})


                print('[{0}/{1}] : {2} -- loss : {3}'.format(self.ccc , int(self.meta_json["NUM_STEPS"]/1000) , np.mean(dap), ls) )
                try:
                    # self.CAMgen(x_test,y_test)
                    save_path = self.saver.save(self.sess, self.save_dir + "/model.ckpt")
                    save_path = self.saver.save(self.sess, self.save_dir + "/model" + str(i) + "iter.ckpt")
                    summary = self.sess.run(merged, feed_dict={self.x_in: x_train[e_bat*j:e_bat*(j+1),:,:,:,:], self.y_: y[e_bat*j:e_bat*(j+1),:]})
                    self.writer.add_summary(summary, i)
                except:
                    print("checkpoint did not saved. pleas be sure that is well saved")

        self.writer.close()

        return 0

    def accuracy_score(self, x_test, y_test):
        ch = np.array([63,62,64,17,10,5,1,61,8,11,2,18,13,9,6,3,59,58,12,60,19,14,57,56,23,15,7,54,53,55])
        channel_index = tuple((ch-1).tolist())
        x_test = x_test.reshape((x_test.shape[0],x_test.shape[1],x_test.shape[2],x_test.shape[3],1))
        x_test = x_test.reshape((x_test.shape[0],x_test.shape[1],x_test.shape[2],x_test.shape[3],1))
        x_trains = x_test[:,channel_index, :, :30, :]
        x_test = np.zeros((x_test.shape[0], 30, 30, x_trains.shape[3], 1)) # ? 64 30 30 1
        for tr in range(x_test.shape[2]):
            x_test[:,:,tr,:,:] = (x_trains[:,:,2*tr,:,:] +x_trains[:,:,2*tr+1,:,:]) /2


        y_mod = np.zeros((y_test.shape[0],2), dtype=np.int16)
        y_test = y_test.astype(np.float32)
        for i in range(y_test.shape[0]):
            y_mod[i, int(((y_test[i]) + 1)/2)] = 1
        # y = tf.Variable(y_mod,dtype=tf.float32)
        y = y_mod.astype(np.float32)

        # y = tf.Variable(y_mod,dtype=tf.float32)
        y__ = y_mod.astype(np.float32)
        y_pred = np.argmax(self.sess.run(self.y, feed_dict={self.x_in: x_test}),axis=1)
        self.acc = np.average(np.equal(y_pred, np.argmax(y__, axis = 1) ))

        return self.acc

class CNN3D_TFC_CAM(CNN3D_TFC):
    def soms(self):
        som = []
        somename=[]
        for f in os.listdir(self.meta_json["MODEL_NAME"] + '-'  + self.meta_json["STARTED_DATESTRING"] + '/acc_sub' + str(self.sub)):
            if fnmatch.fnmatch(f, '*iter.ckpt.meta'):
                som.append(self.meta_json["MODEL_NAME"] + '-'  + self.meta_json["STARTED_DATESTRING"] + '/acc_sub' + str(self.sub) + '/' + f)
                somename.append(f[5:-14])
        return som, somename

    def restoring_variables(self):
        # som = [f for f in os.listdir(self.meta_json["MODEL_NAME"] + '-'  + self.meta_json["STARTED_DATESTRING"] + '/acc_sub' + str(self.sub)) if f.lower().endswith('iter.ckpt')]
        som = []
        for f in os.listdir(self.meta_json["MODEL_NAME"] + '-'  + self.meta_json["STARTED_DATESTRING"] + '/acc_sub' + str(self.sub)):
            if fnmatch.fnmatch(f, '*iter.ckpt.meta'):
                som.append(f)
        self.saver.restore(self.sess, self.meta_json["MODEL_NAME"] + '-'  + self.meta_json["STARTED_DATESTRING"]+ '/acc_sub' + str(self.sub) + '/' + som[-1][:-5])


        # self.saver.restore(self.sess, self.meta_json["MODEL_NAME"] + '-'  + self.meta_json["STARTED_DATESTRING"] + "/model.ckpt")
        self.ccc = 100

        return

    def initialize_variables(self):
        # reshape x_train to 2D image
        # For DNN with 5 channels only : E6(Fz), E10(Fp1), E5(Fp2), E18(F7), E58(F8)
        # For CNN2D with 4 channels only : E10(Fp1), E5(Fp2), E18(F7), E58(F8)
        # x_train = np.zeros((100,64,61,61))
        # shape of the image = (batch, 61*2, 61*2)

        # # controls the learning rate decay.
        # batch = tf.Variable(0)
        #
        # learning_rate = tf.train.exponential_decay(
        #     1e-4,  # Base learning rate.
        #     batch * batch_size,  # Current index into the dataset.
        #     train_size,  # Decay step.
        #     0.95,  # Decay rate.
        #     staircase=True)
        tf.reset_default_graph()


        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        # global_step = tf.train.get_or_create_global_step()
        channel_index = (4, 9, 17, 57)

        # conv network
        self.x_in = tf.placeholder(shape = [None, 120, 121, 121, 1], dtype=tf.float32)
        self.y_ = tf.placeholder(dtype=tf.float32, shape=[None, 2])


        # with tf.device('/cpu:0'):
        w1 = tf.Variable(tf.truncated_normal(shape = (7,7,7,1,10) , stddev=0.1), name="filter1")
        conv1 = tf.nn.conv3d(self.x_in, w1, strides = [1,2,2,2,1], padding="VALID")  # (?, 27, 27, 27, 10)
        conv1_out = tf.nn.relu(conv1) # (?, 116, 116, 16)
        pool1 = tf.nn.max_pool3d(conv1_out, ksize= [1,3,3,3,1], strides=[1,1,1,1,1], padding='SAME')  # (?, 27, 27, 27, 10)

        w11 = tf.Variable(tf.truncated_normal(shape = (7,7,7,10,30) , stddev=0.1), name="filter11")
        conv11 = tf.nn.conv3d(pool1, w11, strides = [1,1,1,1,1], padding="VALID")  # (?, 21, 21, 21, 30)
        # with tf.device('/cpu:0'):
        conv11_out = tf.nn.relu(conv11) # (?, 116, 116, 16)
        self.pool2 = tf.nn.max_pool3d(conv11_out, ksize= [1,3,3,3,1], strides=[1,2,2,2,1], padding='SAME')# (?, 26, 26, 26,30)


        # This model cannot be structurally deep since this model applied 1d convolution

        self.gap = tf.reduce_mean(tf.reshape(self.pool2, shape=(-1, 26*26*26, 30)), axis=1)
        self.weight_ = tf.Variable(tf.zeros((30, 2), dtype=tf.float32), dtype=tf.float32, name="dense_weight")
        self.out_layer = self.gap
        self.y = tf.matmul(self.out_layer, self.weight_)

        # cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.y, labels=self.y_))

        self.train_step = tf.train.AdamOptimizer(learning_rate=self.meta_json["LEARNING_RATE"]).minimize(self.loss)


        # loss_ = self.loss
        # loss_hist = tf.summary.scalar('loss', loss_)
        # self.merged = tf.summary.merge_all()
        # self.writer = tf.summary.FileWriter(
        #     './' + self.meta_json["MODEL_NAME"] + self.meta_json["STARTED_DATESTRING"] + self.meta_json["LOGDIR_ROOT"],
        #     self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()


        return

    # def writeSummary(self):
    #     os.path.join("./"+meta_json["STARTED_DATESTRING"]+meta_json["LOGDIR_ROOT"])
    #     with open(meta_json["MODEL_NAME"] + 'log', "a+") as f:
    #         for i in self.loss_list:
    #             f.write(str(i[0])+","+str(i[1]))
    #         f.write("\n")
    #     os.path.join("..")


    def train(self, x_train, y_train, x_test, y_test):
        ch = np.array([63,62,64,17,10,5,1,61,8,11,2,18,13,9,6,3,59,58,12,60,19,14,57,56,23,15,7,54,53,55])
        channel_index = tuple((ch-1).tolist())
        if self.meta_json["rest"] is True:
            self.initialize_variables()
            self.restoring_variables()
        else:
            self.initialize_variables()

        x_train = x_train.reshape((x_train.shape[0],x_train.shape[1],x_train.shape[2],x_train.shape[3],1))  # (?, 64, 121, 121, 1)
        x_trains = x_train[:, channel_index, :, :, :]
        x_train = np.zeros((x_train.shape[0], 120, 121, 121, 1))
        for i in range(30):
            x_train[:, 4 * i, :, :, :] = x_trains[:, i, :, :, :]
            x_train[:, 4 * i + 1, :, :, :] = x_trains[:, i, :, :, :]
            x_train[:, 4 * i + 2, :, :, :] = x_trains[:, i, :, :, :]
            x_train[:, 4 * i + 3, :, :, :] = x_trains[:, i, :, :, :]


        x_tester = x_test.reshape((x_test.shape[0], x_test.shape[1], x_test.shape[2], x_test.shape[3], 1))
        x_test_inter = x_tester[:, channel_index, :, :, :]
        x_tester = np.zeros((x_test.shape[0], 120, 121, 121, 1))
        for i in range(30):
            x_tester[:, 4 * i, :, :, :] = x_test_inter[:, i, :, :, :]
            x_tester[:, 4 * i + 1, :, :, :] = x_test_inter[:, i, :, :, :]
            x_tester[:, 4 * i + 2, :, :, :] = x_test_inter[:, i, :, :, :]
            x_tester[:, 4 * i + 3, :, :, :] = x_test_inter[:, i, :, :, :]


        y_mod = np.zeros((y_train.shape[0],2), dtype=np.int16)
        y_train = y_train.astype(np.float32)
        for i in range(y_train.shape[0]):
            y_mod[i, int(((y_train[i]) + 1)/2)] = 1
        # y = tf.Variable(y_mod,dtype=tf.float32)
        y = y_mod.astype(np.float32)


        y_mod = np.zeros((y_test.shape[0], 2), dtype=np.int16)
        y_test = y_test.astype(np.float32)
        for i in range(y_test.shape[0]):
            y_mod[i, int(((y_test[i]) + 1) / 2)] = 1
        # y = tf.Variable(y_mod,dtype=tf.float32)
        y_tester = y_mod.astype(np.float32)

        '''
        loss_ = self.loss
        loss_hist = tf.summary.scalar('loss', loss_)
        merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(
            './' + self.meta_json["MODEL_NAME"] + self.meta_json["STARTED_DATESTRING"] + self.meta_json["LOGDIR_ROOT"],
            self.sess.graph)
        '''
        # x_tt = tf.Variable( tf.cast( tf.reshape(x_train, (x_train.shape[0], -1) ), dtype=tf.float32) )
        # y_mod = np.zeros((y_train.shape[0],2), dtype=np.int16)
        # y_train = y_train.astype(np.float32)
        # for i in range(y_train.shape[0]):
        #     y_mod[i, int(((y_train[i]) + 1)/2)] = 1
        # # y = tf.Variable(y_mod,dtype=tf.float32)
        # y = y_mod.astype(np.float32)
        loss_ = self.loss
        loss_hist = tf.summary.scalar('loss', loss_)
        merged = tf.summary.merge_all()

        self.writer = tf.summary.FileWriter(
            './' + self.meta_json["MODEL_NAME"] + self.meta_json["STARTED_DATESTRING"] + self.meta_json[
                "LOGDIR_ROOT"] + '/log_sub{0}_cv{1}'.format(self.sub, self.cv),
            self.sess.graph)

        for i in range(int(self.meta_json["NUM_STEPS"])):
            totbat = x_train.shape[0]
            fortemp = 1
            while (totbat/fortemp) > 100:
                fortemp += 1
            e_bat =int(totbat / fortemp)
            e_batt = int(x_tester.shape[0]/fortemp)
            for j in range(fortemp):
                self.sess.run(self.train_step, feed_dict={self.x_in: x_train[e_bat*j:e_bat*(j+1),:,:,:,:], self.y_: y[e_bat*j:e_bat*(j+1),:]})

            if i%(self.meta_json["CHECKPOINT_EVERY"]) == 0:
                self.ccc += 1
                totbat_ = x_test.shape[0]
                fortemp_ = 1
                while (totbat_ / fortemp_) > 150:
                    fortemp_ += 1
                e_bat_ = int(totbat_ / fortemp_)
                dap = []
                for k in range(fortemp_):
                    dap.append(self.accuracy_score(x_test[e_bat_*k:e_bat_*(k+1),:,:,:], y_test[e_bat_*k:e_bat_*(k+1)]))
                train_loss = 0
                test_loss = 0
                for j in range(fortemp):
                    train_loss += self.sess.run(self.loss, feed_dict={self.x_in: x_train[e_bat*j:e_bat*(j+1),:,:,:,:], self.y_: y[e_bat*j:e_bat*(j+1), :]})
                    test_loss += self.sess.run(self.loss, feed_dict={self.x_in: x_tester[e_batt*j:e_batt*(j+1),:,:,:,:], self.y_: y_tester[e_batt*j:e_batt*(j+1), :]})

                test_acc = self.accuracy_score(x_test, y_test)

                print('[{0}/{1}] : {2} -- test_loss : {3}'.format(self.ccc, int(self.meta_json["NUM_STEPS"]/self.meta_json["CHECKPOINT_EVERY"]), np.mean(dap), test_loss))
                try:

                    self.CAMgen(x_test,y_test)

                    save_path = self.saver.save(self.sess, self.save_dir + "/model0.ckpt")
                    save_path = self.saver.save(self.sess, self.save_dir + "/model" + str(i) + "iter.ckpt")
                    save_path = self.saver.save(self.sess, self.save_dir + '/cv{0}'.format(self.cv) + "/model" + str(i) + "iter.ckpt")

                    summary_train_loss = tf.Summary(value=[tf.Summary.Value(tag="training_batch_loss", simple_value=train_loss)])
                    summary_test_loss = tf.Summary(value=[tf.Summary.Value(tag="test_loss", simple_value=test_loss)])
                    summary_test_acc = tf.Summary(value=[tf.Summary.Value(tag="test_acc", simple_value=test_acc)])
                    self.writer.add_summary(summary_train_loss, i)
                    self.writer.add_summary(summary_test_loss, i)
                    self.writer.add_summary(summary_test_acc, i)
                    # self.loss_list.append((i, ls))
                    # loss_value = self.sess.run(self.out, feed_dict={self.loss_scalar: ls})
                    # summary = self.sess.run(self.summary_op, feed_dict={self.loss_scalar: ls})

                except Exception as e:
                    print(e)
                    print("*"*50)
                    print("checkpoint didn't saved. please be sure that is well saved")
                    print("*" * 50)


        self.writer.close()
        # self.writeSummary()

        return 0

    def CAMgen(self, x_test, y_test):
        # none 12 12 30
        # num_feature = self.weight_.shape[0]
        # num_classes = self.weight_.shape[1]
        ch = np.array([63,62,64,17,10,5,1,61,8,11,2,18,13,9,6,3,59,58,12,60,19,14,57,56,23,15,7,54,53,55])
        channel_index = tuple((ch-1).tolist())
        x_test = x_test.reshape((x_test.shape[0],x_test.shape[1],x_test.shape[2],x_test.shape[3],1))
        x_trains = x_test[:, channel_index, :, :, :]

        x_test = np.zeros((x_test.shape[0], 120, 121, 121, 1))
        for i in range(30):
            x_test[:, 4 * i, :, :, :] = x_trains[:, i, :, :, :]
            x_test[:, 4 * i + 1, :, :, :] = x_trains[:, i, :, :, :]
            x_test[:, 4 * i + 2, :, :, :] = x_trains[:, i, :, :, :]
            x_test[:, 4 * i + 3, :, :, :] = x_trains[:, i, :, :, :]

        y_mod = np.zeros((y_test.shape[0],2), dtype=np.int16)
        y_test = y_test.astype(np.float32)
        for i in range(y_test.shape[0]):
            y_mod[i, int(((y_test[i]) + 1)/2)] = 1

        # y = tf.Variable(y_mod,dtype=tf.float32)
        y = y_mod.astype(np.float32)

        weights_ = self.sess.run(self.weight_)  # 30,2
        pool2seed = self.sess.run(self.pool2, feed_dict={self.x_in: x_test})  # batch,26,26,26,30
        self.image = np.zeros((26, 26, 26, 2))


        for bi in range(pool2seed.shape[0]):
            for i in range(self.weight_.shape[0]):
                self.image[:, :, :, np.where(y[bi] == 1)[0].tolist()[0]] += (pool2seed[bi,:,:,:,:].reshape((26,26,26,30))[:,:,:,i] * weights_[i, np.where(y[bi] == 1)[0].tolist()[0]])

        # divide im 1000
        self.image = self.image - np.min(self.image)
        denom = np.max(self.image)/1000
        self.image = np.int16(self.image/denom)

        np.save(self.save_dir + '/CAM_{0}'.format(self.ccc), self.image)
        return 0

    def accuracy_score(self, x_test, y_test):
        ch = np.array([63,62,64,17,10,5,1,61,8,11,2,18,13,9,6,3,59,58,12,60,19,14,57,56,23,15,7,54,53,55])
        channel_index = tuple((ch-1).tolist())
        x_test = x_test.reshape((x_test.shape[0],x_test.shape[1],x_test.shape[2],x_test.shape[3],1))
        x_trains = x_test[:,channel_index, :, :, :]

        x_test = np.zeros((x_test.shape[0], 120, 121, 121, 1))
        for i in range(30):
            x_test[:, 4 * i, :, :, :] = x_trains[:, i, :, :, :]
            x_test[:, 4 * i + 1, :, :, :] = x_trains[:, i, :, :, :]
            x_test[:, 4 * i + 2, :, :, :] = x_trains[:, i, :, :, :]
            x_test[:, 4 * i + 3, :, :, :] = x_trains[:, i, :, :, :]


        y_mod = np.zeros((y_test.shape[0],2), dtype=np.int16)
        y_test = y_test.astype(np.float32)
        for i in range(y_test.shape[0]):
            y_mod[i, int(((y_test[i]) + 1)/2)] = 1
        # y = tf.Variable(y_mod,dtype=tf.float32)
        y = y_mod.astype(np.float32)

        # y = tf.Variable(y_mod,dtype=tf.float32)
        y__ = y_mod.astype(np.float32)
        y_pred = np.argmax(self.sess.run(self.y, feed_dict={self.x_in: x_test}), axis=1)
        self.acc = np.average(np.equal(y_pred, np.argmax(y__, axis=1)))

        return self.acc

class CNN3DLSTM(CNN3D_TFC_CAM):
    def __init__(self, meta_json, save_dir, sub=0, cv=1):
        input_layer = 3
        self.meta_json = meta_json
        self.meta_json['addto'] = self.meta_json['MODEL_NAME']
        self.ccc = 0
        self.save_dir = save_dir
        self.sub = sub +1
        self.cv = cv
        self.build_addon()
        self.puts_out_tw = 20 # the length of the time window of return values of the puts_out method.

        return
    def restoring_variables(self):
        # som = [f for f in os.listdir(self.meta_json["MODEL_NAME"] + '-'  + self.meta_json["STARTED_DATESTRING"] + '/acc_sub' + str(self.sub)) if f.lower().endswith('iter.ckpt')]
        som = []
        for f in os.listdir(self.meta_json["MODEL_NAME"] + '-'  + self.meta_json["STARTED_DATESTRING"] + '/acc_sub' + str(self.sub)):
            if fnmatch.fnmatch(f, '*iter.ckpt.meta'):
                som.append(f)
        self.saver.restore(self.sess, self.meta_json["MODEL_NAME"] + '-'  + self.meta_json["STARTED_DATESTRING"]+ '/acc_sub' + str(self.sub) + '/' + som[-1][:-5])


        # self.saver.restore(self.sess, self.meta_json["MODEL_NAME"] + '-'  + self.meta_json["STARTED_DATESTRING"] + "/model.ckpt")
        self.ccc = 100

        return

    def initialize_variables(self):
        tf.reset_default_graph()
        initializer = tf.contrib.layers.xavier_initializer()

        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        channel_index = (4, 9, 17, 57)

        # conv network
        self.x_in = tf.placeholder(shape = [None, 120, 121, 121, 1], dtype=tf.float32)
        self.y_ = tf.placeholder(dtype=tf.float32, shape=[None, 2])


        # with tf.device('/cpu:0'):
        w1 = tf.Variable(tf.truncated_normal(shape = (7,7,7,1,10) , stddev=0.1), name="filter1")
        conv1 = tf.nn.conv3d(self.x_in, w1, strides = [1,2,2,2,1], padding="VALID")  # (?, 27, 27, 27, 10)
        conv1_out = tf.nn.relu(conv1) # (?, 116, 116, 16)
        pool1 = tf.nn.max_pool3d(conv1_out, ksize= [1,3,3,3,1], strides=[1,1,1,1,1], padding='SAME')  # (?, 27, 27, 27, 10)

        w11 = tf.Variable(tf.truncated_normal(shape = (7,7,7,10,30) , stddev=0.1), name="filter11")
        conv11 = tf.nn.conv3d(pool1, w11, strides = [1,1,1,1,1], padding="VALID")  # (?, 21, 21, 21, 30)
        # with tf.device('/cpu:0'):
        conv11_out = tf.nn.relu(conv11) # (?, 116, 116, 16)
        self.pool2 = tf.nn.max_pool3d(conv11_out, ksize= [1,3,3,3,1], strides=[1,2,2,2,1], padding='SAME')# (?, 26, 26, 26,30)


        # This model cannot be structurally deep since this model applied 1d convolution

        self.gap = tf.reduce_mean(tf.reshape(self.pool2, shape=(-1, 26*26*26, 30)), axis=1)
        self.weight_ = tf.Variable(tf.zeros((30, 2), dtype=tf.float32), dtype=tf.float32, name="dense_weight")
        self.out_layer = self.gap
        self.y = tf.matmul(self.out_layer, self.weight_)


        # cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
        # self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.y, labels=self.y_))
        self.towardLSTM = tf.argmax(self.y,axis=1)

        # loss_ = self.loss
        # loss_hist = tf.summary.scalar('loss', loss_)
        # self.merged = tf.summary.merge_all()
        # self.writer = tf.summary.FileWriter(
        #     './' + self.meta_json["MODEL_NAME"] + self.meta_json["STARTED_DATESTRING"] + self.meta_json["LOGDIR_ROOT"],
        #     self.sess.graph)

        meta_json = self.meta_json
        learning_rate = meta_json["LEARNING_RATE"]
        total_epoch = meta_json["NUM_STEPS"]
        batch_size = meta_json["SAMPLE_SIZE"]

        ## TODO: implement bidirectional LSTM

        n_input = 3  # state, goal, MB/MF
        n_step = self.puts_out_tw
        n_hidden = 100
        d_a_size = 200
        r_size = 50
        fc_size = 500
        num_classes = 2
        p_coef = 1.0

        self.X2 = tf.placeholder(tf.float32, [None, self.puts_out_tw, 2])
        self.X2 = tf.placeholder(tf.float32, [None, 2])

        specs = []
        self.X = tf.zeros(size=[None, self.puts_out_tw-1 , 3])





        
        specs = np.array(specs)
        Z_ = np.concatenate((specs, np.zeros((model1.puts_out_tw - 1, specs.shape[1]))), axis=0)
        X_tt = []
        for xtpi in range(specs.shape[0]):

            X_tt.append(Z_[xtpi:xtpi + model1.puts_out_tw, :])
        xtt = np.array(X_tt)
        X_tt = []

        self.X = tf.concat([self.towardLSTM, self.X2], axis=0)
        self.Y = tf.placeholder(tf.float32, [None, num_classes])

        # text_length = self._length(self.input_text) # size = batch_size I dont think we need this argument

        # Bidirectional(Left&Right) Recurrent Structure
        with tf.name_scope("bi-lstm"):
            fw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
            bw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
            (self.output_fw, self.output_bw), states = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell,
                                                                                       cell_bw=bw_cell,
                                                                                       inputs=self.X,
                                                                                       dtype=tf.float32)
            self.H = tf.concat([self.output_fw, self.output_bw], axis=2)
            H_reshape = tf.reshape(self.H, [-1, 2 * n_hidden])

        with tf.name_scope("self-attention"):
            self.W_s1 = tf.get_variable("W_s1", shape=[2 * n_hidden, d_a_size], initializer=initializer)
            _H_s1 = tf.nn.tanh(tf.matmul(H_reshape, self.W_s1))
            self.W_s2 = tf.get_variable("W_s2", shape=[d_a_size, r_size], initializer=initializer)
            _H_s2 = tf.matmul(_H_s1, self.W_s2)
            _H_s2_reshape = tf.transpose(tf.reshape(_H_s2, [-1, self.puts_out_tw, r_size]), [0, 2, 1])
            self.A = tf.nn.softmax(_H_s2_reshape, name="attention")

        with tf.name_scope("sentence-embedding"):
            self.M = tf.matmul(self.A, self.H)

        with tf.name_scope("fully-connected"):
            # self.M_pool = tf.reduce_mean(self.M, axis=1)
            # W_fc = tf.get_variable("W_fc", shape=[2 * hidden_size, fc_size], initializer=initializer)
            self.M_flat = tf.reshape(self.M, shape=[-1, 2 * n_hidden * r_size])
            W_fc = tf.get_variable("W_fc", shape=[2 * n_hidden * r_size, fc_size], initializer=initializer)
            b_fc = tf.Variable(tf.constant(0.1, shape=[fc_size]), name="b_fc")
            self.fc = tf.nn.relu(tf.nn.xw_plus_b(self.M_flat, W_fc, b_fc), name="fc")

        with tf.name_scope("output"):
            W_output = tf.get_variable("W_output", shape=[fc_size, num_classes], initializer=initializer)
            b_output = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b_output")
            self.logits = tf.nn.xw_plus_b(self.fc, W_output, b_output, name="logits")
            self.predictions = tf.argmax(self.logits, 1, name="predictions")

        with tf.name_scope("penalization"):
            self.AA_T = tf.matmul(self.A, tf.transpose(self.A, [0, 2, 1]))
            self.I = tf.reshape(tf.tile(tf.eye(r_size), [tf.shape(self.A)[0], 1]), [-1, r_size, r_size])
            self.P = tf.square(tf.norm(self.AA_T - self.I, axis=[-2, -1], ord="fro"))

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y)
            self.loss_P = tf.reduce_mean(self.P * p_coef)
            self.loss = tf.reduce_mean(losses) + self.loss_P

        with tf.name_scope("optimizer"):
            self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.Y, axis=1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name="accuracy")
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

        return

    def train(self, x_train, x_train2, y_train, x_test, x_test2, y_test):
        ch = np.array([63,62,64,17,10,5,1,61,8,11,2,18,13,9,6,3,59,58,12,60,19,14,57,56,23,15,7,54,53,55])
        channel_index = tuple((ch-1).tolist())
        if self.meta_json["rest"] is True:
            self.initialize_variables()
            self.restoring_variables()
        else:
            self.initialize_variables()

        x_train = x_train.reshape((x_train.shape[0],x_train.shape[1],x_train.shape[2],x_train.shape[3],1))  # (?, 64, 121, 121, 1)
        x_trains = x_train[:, channel_index, :, :, :]
        x_train = np.zeros((x_train.shape[0], 120, 121, 121, 1))
        for i in range(30):
            x_train[:, 4 * i, :, :, :] = x_trains[:, i, :, :, :]
            x_train[:, 4 * i + 1, :, :, :] = x_trains[:, i, :, :, :]
            x_train[:, 4 * i + 2, :, :, :] = x_trains[:, i, :, :, :]
            x_train[:, 4 * i + 3, :, :, :] = x_trains[:, i, :, :, :]


        x_tester = x_test.reshape((x_test.shape[0], x_test.shape[1], x_test.shape[2], x_test.shape[3], 1))
        x_test_inter = x_tester[:, channel_index, :, :, :]
        x_tester = np.zeros((x_test.shape[0], 120, 121, 121, 1))
        for i in range(30):
            x_tester[:, 4 * i, :, :, :] = x_test_inter[:, i, :, :, :]
            x_tester[:, 4 * i + 1, :, :, :] = x_test_inter[:, i, :, :, :]
            x_tester[:, 4 * i + 2, :, :, :] = x_test_inter[:, i, :, :, :]
            x_tester[:, 4 * i + 3, :, :, :] = x_test_inter[:, i, :, :, :]


        y_mod = np.zeros((y_train.shape[0],2), dtype=np.int16)
        y_train = y_train.astype(np.float32)
        for i in range(y_train.shape[0]):
            y_mod[i, int(((y_train[i]) + 1)/2)] = 1
        # y = tf.Variable(y_mod,dtype=tf.float32)
        y = y_mod.astype(np.float32)


        y_mod = np.zeros((y_test.shape[0], 2), dtype=np.int16)
        y_test = y_test.astype(np.float32)
        for i in range(y_test.shape[0]):
            y_mod[i, int(((y_test[i]) + 1) / 2)] = 1
        # y = tf.Variable(y_mod,dtype=tf.float32)
        y_tester = y_mod.astype(np.float32)

        '''
        loss_ = self.loss
        loss_hist = tf.summary.scalar('loss', loss_)
        merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(
            './' + self.meta_json["MODEL_NAME"] + self.meta_json["STARTED_DATESTRING"] + self.meta_json["LOGDIR_ROOT"],
            self.sess.graph)
        '''
        # x_tt = tf.Variable( tf.cast( tf.reshape(x_train, (x_train.shape[0], -1) ), dtype=tf.float32) )
        # y_mod = np.zeros((y_train.shape[0],2), dtype=np.int16)
        # y_train = y_train.astype(np.float32)
        # for i in range(y_train.shape[0]):
        #     y_mod[i, int(((y_train[i]) + 1)/2)] = 1
        # # y = tf.Variable(y_mod,dtype=tf.float32)
        # y = y_mod.astype(np.float32)
        loss_ = self.loss
        loss_hist = tf.summary.scalar('loss', loss_)
        merged = tf.summary.merge_all()

        self.writer = tf.summary.FileWriter(
            './' + self.meta_json["MODEL_NAME"] + self.meta_json["STARTED_DATESTRING"] + self.meta_json[
                "LOGDIR_ROOT"] + '/log_sub{0}_cv{1}'.format(self.sub, self.cv),
            self.sess.graph)

        for i in range(int(self.meta_json["NUM_STEPS"])):
            totbat = x_train.shape[0]
            fortemp = 1
            while (totbat/fortemp) > 100:
                fortemp += 1
            e_bat =int(totbat / fortemp)
            e_batt = int(x_tester.shape[0]/fortemp)
            for j in range(fortemp):
                self.sess.run(self.train_step, feed_dict={self.x_in: x_train[e_bat*j:e_bat*(j+1),:,:,:,:], self.y_: y[e_bat*j:e_bat*(j+1),:]})

            if i%(self.meta_json["CHECKPOINT_EVERY"]) == 0:
                self.ccc += 1
                totbat_ = x_test.shape[0]
                fortemp_ = 1
                while (totbat_ / fortemp_) > 150:
                    fortemp_ += 1
                e_bat_ = int(totbat_ / fortemp_)
                dap = []
                for k in range(fortemp_):
                    dap.append(self.accuracy_score(x_test[e_bat_*k:e_bat_*(k+1),:,:,:], y_test[e_bat_*k:e_bat_*(k+1)]))
                train_loss = 0
                test_loss = 0
                for j in range(fortemp):
                    train_loss += self.sess.run(self.loss, feed_dict={self.x_in: x_train[e_bat*j:e_bat*(j+1),:,:,:,:], self.y_: y[e_bat*j:e_bat*(j+1), :]})
                    test_loss += self.sess.run(self.loss, feed_dict={self.x_in: x_tester[e_batt*j:e_batt*(j+1),:,:,:,:], self.y_: y_tester[e_batt*j:e_batt*(j+1), :]})

                test_acc = self.accuracy_score(x_test, y_test)

                print('[{0}/{1}] : {2} -- test_loss : {3}'.format(self.ccc, int(self.meta_json["NUM_STEPS"]/self.meta_json["CHECKPOINT_EVERY"]), np.mean(dap), test_loss))
                try:

                    self.CAMgen(x_test,y_test)

                    save_path = self.saver.save(self.sess, self.save_dir + "/model0.ckpt")
                    save_path = self.saver.save(self.sess, self.save_dir + "/model" + str(i) + "iter.ckpt")
                    save_path = self.saver.save(self.sess, self.save_dir + '/cv{0}'.format(self.cv) + "/model" + str(i) + "iter.ckpt")

                    summary_train_loss = tf.Summary(value=[tf.Summary.Value(tag="training_batch_loss", simple_value=train_loss)])
                    summary_test_loss = tf.Summary(value=[tf.Summary.Value(tag="test_loss", simple_value=test_loss)])
                    summary_test_acc = tf.Summary(value=[tf.Summary.Value(tag="test_acc", simple_value=test_acc)])
                    self.writer.add_summary(summary_train_loss, i)
                    self.writer.add_summary(summary_test_loss, i)
                    self.writer.add_summary(summary_test_acc, i)
                    # self.loss_list.append((i, ls))
                    # loss_value = self.sess.run(self.out, feed_dict={self.loss_scalar: ls})
                    # summary = self.sess.run(self.summary_op, feed_dict={self.loss_scalar: ls})

                except Exception as e:
                    print(e)
                    print("*"*50)
                    print("checkpoint didn't saved. please be sure that is well saved")
                    print("*" * 50)


        self.writer.close()
        # self.writeSummary()

        return 0


class addon_bidi_LSTM:
    # origianl input data size : 61x61x64(frequency x time point x channel number) to (datasize, time) shape of data
    # the data returned by the puts_out method are different from the return value of the puts_out in the addon_LSTM class.
    # It should has the temporal window for the history.
    # reference : A novel deep learning approach for classification of EEG motor imagerty signals, Yousef Razael Tabar and Ugur Halici, 2017, Journal of Neural Engineering
    #
    def __init__(self, meta_json, save_dir, sub=0, cv=1):
        input_layer = 3
        self.meta_json = meta_json
        self.meta_json['addto'] = self.meta_json['MODEL_NAME']
        self.ccc = 0
        self.save_dir = save_dir
        self.sub = sub +1
        self.cv = cv
        self.build_addon()
        self.puts_out_tw = 20 # the length of the time window of return values of the puts_out method.

        return

    def build_addon(self):
        if self.meta_json['addto'] == 'nothing':
            print('BOOM in half')
        else:
            self.model = eval(self.meta_json['addto'] + '(self.meta_json, self.save_dir, self.sub, self.cv)')
            self.model.initialize_variables()
            # self.model.restoring_variables()
        return

    def puts_out(self, spectro_train, spectro_test):
        if self.meta_json['addto'] == 'CNN3D_TFC_CAM':

            ch = np.array(
                [63, 62, 64, 17, 10, 5, 1, 61, 8, 11, 2, 18, 13, 9, 6, 3, 59, 58, 12, 60, 19, 14, 57, 56, 23, 15, 7, 54,
                 53, 55])
            channel_index = tuple((ch - 1).tolist())

            x_train = spectro_train.reshape(
                (spectro_train.shape[0], spectro_train.shape[1], spectro_train.shape[2], spectro_train.shape[3], 1))  # (?, 64, 121, 121, 1)
            x_trains = x_train[:, channel_index, :, :, :]
            x_train = np.zeros((x_train.shape[0], 120, 121, 121, 1))
            for i in range(30):
                x_train[:, 4 * i, :, :, :] = x_trains[:, i, :, :, :]
                x_train[:, 4 * i + 1, :, :, :] = x_trains[:, i, :, :, :]
                x_train[:, 4 * i + 2, :, :, :] = x_trains[:, i, :, :, :]
                x_train[:, 4 * i + 3, :, :, :] = x_trains[:, i, :, :, :]

            x_tester = spectro_test.reshape((spectro_test.shape[0], spectro_test.shape[1], spectro_test.shape[2], spectro_test.shape[3], 1))
            x_test_inter = x_tester[:, channel_index, :, :, :]
            x_tester = np.zeros((spectro_test.shape[0], 120, 121, 121, 1))
            for i in range(30):
                x_tester[:, 4 * i, :, :, :] = x_test_inter[:, i, :, :, :]
                x_tester[:, 4 * i + 1, :, :, :] = x_test_inter[:, i, :, :, :]
                x_tester[:, 4 * i + 2, :, :, :] = x_test_inter[:, i, :, :, :]
                x_tester[:, 4 * i + 3, :, :, :] = x_test_inter[:, i, :, :, :]
            x_test = x_tester

            y_train = np.argmax(self.model.sess.run(self.model.y,feed_dict={self.model.x_in : x_train}),1)
            y_test = np.argmax(self.model.sess.run(self.model.y,feed_dict={self.model.x_in : x_test}),1)
            return (y_train, y_test)


        elif self.meta_json['addto'] == 'CNN2DwithCAM':
            # self.model.saver.restore(self.model.sess, self.meta_json["MODEL_NAME"] + '-' + self.meta_json[
            #     "STARTED_DATESTRING"] + "/model.ckpt")
            temp1 = np.concatenate((spectro_train[:, 4, :, :], spectro_train[:, 9, :, :]), axis=1)
            temp2 = np.concatenate((spectro_train[:, 17, :, :], spectro_train[:, 57, :, :]), axis=1)
            x_train = np.concatenate((temp1, temp2), axis=2).reshape((spectro_train.shape[0], 121 * 2, 121 * 2, 1))
            temp1 = np.concatenate((spectro_test[:, 4, :, :], spectro_test[:, 9, :, :]), axis=1)
            temp2 = np.concatenate((spectro_test[:, 17, :, :], spectro_test[:, 57, :, :]), axis=1)
            x_test = np.concatenate((temp1, temp2), axis=2).reshape((spectro_test.shape[0], 121 * 2, 121 * 2, 1))

            y_train = np.argmax(self.model.sess.run(self.model.y,feed_dict={self.model.x_in : x_train}),1)
            y_test = np.argmax(self.model.sess.run(self.model.y,feed_dict={self.model.x_in : x_test}),1)
            return (y_train, y_test)

        else:
            y_train = np.random.randint(2, size=(spectro_train.shape[0],))
            y_test = np.random.randint(2, size=(spectro_train.shape[0],))

            return (y_train, y_test)

    def init_LSTM(self):
        tf.reset_default_graph()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        meta_json = self.meta_json
        learning_rate = meta_json["LEARNING_RATE"]
        total_epoch = meta_json["NUM_STEPS"]
        batch_size = meta_json["SAMPLE_SIZE"]

        ## TODO: implement bidirectional LSTM

        n_input = 3 # state, goal, MB/MF
        n_step = self.puts_out_tw
        n_hidden = 100
        d_a_size = 200
        r_size = 50
        fc_size = 500
        num_classes = 2
        p_coef = 1.0

        self.X = tf.placeholder(tf.float32, [None, self.puts_out_tw, n_input])
        self.Y = tf.placeholder(tf.float32, [None, num_classes])

        # text_length = self._length(self.input_text) # size = batch_size I dont think we need this argument
        initializer = tf.contrib.layers.xavier_initializer()

        # Bidirectional(Left&Right) Recurrent Structure
        with tf.name_scope("bi-lstm"):
            fw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
            bw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
            (self.output_fw, self.output_bw), states = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell,
                                                                                       cell_bw=bw_cell,
                                                                                       inputs=self.X,
                                                                                       dtype=tf.float32)
            self.H = tf.concat([self.output_fw, self.output_bw], axis=2)
            H_reshape = tf.reshape(self.H, [-1, 2 * n_hidden])

        with tf.name_scope("self-attention"):
            self.W_s1 = tf.get_variable("W_s1", shape=[2*n_hidden, d_a_size], initializer=initializer)
            _H_s1 = tf.nn.tanh(tf.matmul(H_reshape, self.W_s1))
            self.W_s2 = tf.get_variable("W_s2", shape=[d_a_size, r_size], initializer=initializer)
            _H_s2 = tf.matmul(_H_s1, self.W_s2)
            _H_s2_reshape = tf.transpose(tf.reshape(_H_s2, [-1, self.puts_out_tw, r_size]), [0, 2, 1])
            self.A = tf.nn.softmax(_H_s2_reshape, name="attention")

        with tf.name_scope("sentence-embedding"):
            self.M = tf.matmul(self.A, self.H)

        with tf.name_scope("fully-connected"):
            # self.M_pool = tf.reduce_mean(self.M, axis=1)
            # W_fc = tf.get_variable("W_fc", shape=[2 * hidden_size, fc_size], initializer=initializer)
            self.M_flat = tf.reshape(self.M, shape=[-1, 2 * n_hidden * r_size])
            W_fc = tf.get_variable("W_fc", shape=[2 * n_hidden * r_size, fc_size], initializer=initializer)
            b_fc = tf.Variable(tf.constant(0.1, shape=[fc_size]), name="b_fc")
            self.fc = tf.nn.relu(tf.nn.xw_plus_b(self.M_flat, W_fc, b_fc), name="fc")

        with tf.name_scope("output"):
            W_output = tf.get_variable("W_output", shape=[fc_size, num_classes], initializer=initializer)
            b_output = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b_output")
            self.logits = tf.nn.xw_plus_b(self.fc, W_output, b_output, name="logits")
            self.predictions = tf.argmax(self.logits, 1, name="predictions")

        with tf.name_scope("penalization"):
            self.AA_T = tf.matmul(self.A, tf.transpose(self.A, [0, 2, 1]))
            self.I = tf.reshape(tf.tile(tf.eye(r_size), [tf.shape(self.A)[0], 1]), [-1, r_size, r_size])
            self.P = tf.square(tf.norm(self.AA_T - self.I, axis=[-2, -1], ord="fro"))

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y)
            self.loss_P = tf.reduce_mean(self.P * p_coef)
            self.loss = tf.reduce_mean(losses) + self.loss_P

        with tf.name_scope("optimizer"):
            self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.Y, axis=1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name="accuracy")
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def restoring_variables(self):

        som = []
        for f in os.listdir(self.meta_json["MODEL_NAME"] + '-'  + self.meta_json["STARTED_DATESTRING"]  + '-CNNLSTM' + '/acc_sub' + str(self.sub)):
            if fnmatch.fnmatch(f, '*iter_lstm.ckpt.meta'):
                som.append(f)
        self.saver.restore(self.sess, self.meta_json["MODEL_NAME"] + '-' + self.meta_json["STARTED_DATESTRING"]   + '-CNNLSTM' + '/acc_sub' + str(self.sub) + "/model_lstm.ckpt")

        self.ccc = 100

        return

    def train(self, x_train, y_train, x_test, y_test):
        # x_train, x_test : (number of trials , number of input, number of step)
        # y_train, y_test : (number of trials , 2)

        loss_ = self.loss
        loss_hist = tf.summary.scalar('loss', loss_)
        merged = tf.summary.merge_all()

        try:
            os.mkdir('./' + self.meta_json["MODEL_NAME"] + self.meta_json["STARTED_DATESTRING"]  + '-CNNLSTM' + self.meta_json["LOGDIR_ROOT"] + '/log_sub' + str(self.sub) )
        except:
            print('something wrong while making directory!')

        self.writer = tf.summary.FileWriter(
            './' + self.meta_json["MODEL_NAME"] + self.meta_json["STARTED_DATESTRING"]  + '-CNNLSTM' + self.meta_json["LOGDIR_ROOT"] + '/log_sub'+ str(self.sub) + '/' + '/log_sub'+ str(self.sub) + '_cv' + str(self.cv) ,
            self.sess.graph)

        x_tt = x_train.reshape((x_train.shape[0], -1, 3) ).astype(np.float32)
        # x_tt = tf.Variable( tf.cast( tf.reshape(x_train, (x_train.shape[0], -1) ), dtype=tf.float32) )
        y_mod = np.zeros((y_train.shape[0],2), dtype=np.int16)
        y_train = y_train.astype(np.float32)
        for i in range(y_train.shape[0]):
            y_mod[i, int(((y_train[i]) + 1)/2)] = 1
        # y = tf.Variable(y_mod,dtype=tf.float32)
        y = y_mod.astype(np.float32)

        for i in range(int(self.meta_json["NUM_STEPS"])):
            self.sess.run(self.optimizer , feed_dict = {self.X: x_tt, self.Y: y})
            if i%(self.meta_json["CHECKPOINT_EVERY"]) == 0:
                self.ccc += 1
                print('[{0}/{1}] : {2} -- loss : {3}'.format(self.ccc , int(self.meta_json["NUM_STEPS"]/self.meta_json["CHECKPOINT_EVERY"]) , self.accuracy_score( x_test, y_test), self.sess.run(self.loss, feed_dict = {self.X: x_tt, self.Y: y}) ) )
                try:
                    try:
                        os.mkdir(self.meta_json["MODEL_NAME"] + '-'  + self.meta_json["STARTED_DATESTRING"] + '-CNNLSTM'+ '/acc_sub' + str(self.sub) + "/cv"  + str(self.cv))
                    except:
                        print('')
                    save_path = self.saver.save(self.sess,  self.meta_json["MODEL_NAME"] + '-'  + self.meta_json["STARTED_DATESTRING"] + '-CNNLSTM'+ '/acc_sub' + str(self.sub) + "/model_lstm.ckpt")
                    save_path = self.saver.save(self.sess,  self.meta_json["MODEL_NAME"] + '-'  + self.meta_json["STARTED_DATESTRING"] + '-CNNLSTM'+ '/acc_sub' + str(self.sub) + "/model" + str(i) + "iter_lstm.ckpt")
                    save_path = self.saver.save(self.sess,  self.meta_json["MODEL_NAME"] + '-'  + self.meta_json["STARTED_DATESTRING"] + '-CNNLSTM'+ '/acc_sub' + str(self.sub) + "/cv"  + str(self.cv) + "/model" + str(i) + "iter_lstm.ckpt")
                    self.gen_attmap(x_test, y_test)
                    test_acc = self.sess.run(merged, feed_dict = {self.X: x_tt, self.Y: y})
                    train_loss = self.sess.run(self.loss, feed_dict = {self.X: x_tt, self.Y: y})
                    summary_train_loss = tf.Summary(value=[tf.Summary.Value(tag="training_batch_loss", simple_value=train_loss)])
                    summary_test_acc = tf.Summary(value=[tf.Summary.Value(tag="test_acc", simple_value=test_acc)])
                    self.writer.add_summary(summary_train_loss, i)
                    self.writer.add_summary(summary_test_acc, i)

                except:
                    print("checkpoint didnt saved. pleas be sure that is well saved")

        self.writer.close()

        return 0

    def accuracy_score(self, x_test, y_test):
        x_tt = x_test.reshape((x_test.shape[0], -1, 3) ).astype(np.float32)
        y_mod = np.zeros((y_test.shape[0],2), dtype=np.int16)
        y_train = y_test.astype(np.float32)
        for i in range(y_train.shape[0]):
            y_mod[i, int(((y_train[i]) + 1)/2)] = 1
        # y = tf.Variable(y_mod,dtype=tf.float32)
        y = y_mod.astype(np.float32)
        self.acc = self.sess.run(self.accuracy, feed_dict={self.X :x_tt, self.Y:y})

        return self.acc

    def gen_attmap(self, x_test, y_test):
        x_tt = x_test.reshape((x_test.shape[0], -1, 3) ).astype(np.float32)
        y_mod = np.zeros((y_test.shape[0],2), dtype=np.int16)
        y_train = y_test.astype(np.float32)
        for i in range(y_train.shape[0]):
            y_mod[i, int(((y_train[i]) + 1)/2)] = 1
        # y = tf.Variable(y_mod,dtype=tf.float32)
        y = y_mod.astype(np.float32)
        self._Att_map = self.sess.run(self.A, feed_dict={self.X: x_tt, self.Y: y})
        self.image = np.mean(self._Att_map, axis = 1)
        np.save(self.meta_json["MODEL_NAME"] + '-' + self.meta_json["STARTED_DATESTRING"]  + '-CNNLSTM' + '/acc_sub' + str(self.sub) +  '/ATTMAP_{0}'.format(self.ccc) ,self.image)

        return 0

    def ret_attmap(self, x_test, y_test):
        x_tt = x_test.reshape((x_test.shape[0], -1, 3) ).astype(np.float32)
        y_mod = np.zeros((y_test.shape[0],2), dtype=np.int16)
        y_train = y_test.astype(np.float32)
        for i in range(y_train.shape[0]):
            y_mod[i, int(((y_train[i]) + 1)/2)] = 1
        # y = tf.Variable(y_mod,dtype=tf.float32)
        y = y_mod.astype(np.float32)
        self._Att_map = self.sess.run(self.A, feed_dict={self.X: x_tt, self.Y: y})
        self.image = np.mean(self._Att_map, axis = 1)
        return self.image


class addon_LSTM:
    # origianl input data size : 61x61x64(frequency x time point x channel number) to 1D image
    # reference : A novel deep learning approach for classification of EEG motor imagerty signals, Yousef Razael Tabar and Ugur Halici, 2017, Journal of Neural Engineering
    #
    def __init__(self, meta_json):
        input_layer = 3
        self.meta_json = meta_json
        self.meta_json['addto'] = self.meta_json['MODEL_NAME']
        self.build_addon()
        self.ccc = 0

        return

    def build_addon(self):
        if self.meta_json['addto'] == 'nothing':
            print('BOOM in half')
        else:
            self.model = eval(self.meta_json['addto'] + '(self.meta_json)')
            self.model.initialize_variables()
            self.model.restoring_variables()
        return

    def puts_out(self, spectro_train, spectro_test):
        if self.meta_json['addto'] == 'CNN3D_TFC_CAM':
            ch = np.array([63,62,64,17,10,5,1,61,8,11,2,18,13,9,6,3,59,58,12,60,19,14,57,56,23,15,7,54,53,55])
            channel_index = tuple((ch-1).tolist())
            x_train = spectro_train.reshape((spectro_train.shape[0],spectro_train.shape[1],spectro_train.shape[2],spectro_train.shape[3],1))
            x_trains = x_train[:,channel_index, :, :30, :]
            x_train = np.zeros((x_train.shape[0], 30, 30, x_trains.shape[3], 1)) # ? 64 30 30 1
            for tr in range(x_train.shape[2]):
                x_train[:,:,tr,:,:] = (x_trains[:,:,2*tr,:,:] +x_trains[:,:,2*tr+1,:,:]) /2

            x_test = spectro_test.reshape((spectro_test.shape[0], spectro_test.shape[1], spectro_test.shape[2], spectro_test.shape[3], 1))
            x_trains = x_test[:, channel_index, :, :30, :]
            x_test = np.zeros((x_test.shape[0], 30, 30, x_trains.shape[3], 1))  # ? 64 30 30 1
            for tr in range(x_test.shape[2]):
                x_test[:, :, tr, :, :] = (x_trains[:, :, 2 * tr, :, :] + x_trains[:, :, 2 * tr + 1, :, :]) / 2

            y_train = np.argmax(self.model.sess.run(self.model.y,feed_dict={self.model.x_in : x_train}),1)
            y_test = np.argmax(self.model.sess.run(self.model.y,feed_dict={self.model.x_in : x_test}),1)


            return (y_train, y_test)
        elif self.meta_json['addto'] == 'CNN2DwithCAM':
            # self.model.saver.restore(self.model.sess, self.meta_json["MODEL_NAME"] + '-' + self.meta_json[
            #     "STARTED_DATESTRING"] + "/model.ckpt")
            temp1 = np.concatenate((spectro_train[:, 4, :, :], spectro_train[:, 9, :, :]), axis=1)
            temp2 = np.concatenate((spectro_train[:, 17, :, :], spectro_train[:, 57, :, :]), axis=1)
            x_train = np.concatenate((temp1, temp2), axis=2).reshape((spectro_train.shape[0], 61 * 2, 61 * 2, 1))
            temp1 = np.concatenate((spectro_test[:, 4, :, :], spectro_test[:, 9, :, :]), axis=1)
            temp2 = np.concatenate((spectro_test[:, 17, :, :], spectro_test[:, 57, :, :]), axis=1)
            x_test = np.concatenate((temp1, temp2), axis=2).reshape((spectro_test.shape[0], 61 * 2, 61 * 2, 1))

            y_train = np.argmax(self.model.sess.run(self.model.y,feed_dict={self.model.x_in : x_train}),1)
            y_test = np.argmax(self.model.sess.run(self.model.y,feed_dict={self.model.x_in : x_test}),1)


            return (y_train, y_test)
        else:
            print('ERROR>????')
            y_train = np.zeros((spectro_train.shape[0],))
            y_test = np.zeros((spectro_test.shape[0],))

            return (y_train, y_test)

    def init_LSTM(self):
        tf.reset_default_graph()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        meta_json = self.meta_json
        learning_rate = meta_json["LEARNING_RATE"]
        total_epoch = meta_json["NUM_STEPS"]
        batch_size = meta_json["SAMPLE_SIZE"]



        n_input = 3 # state, goal, MB/MF
        n_step = 1
        n_hidden = 128
        n_class = 2

        self.X = tf.placeholder(tf.float32, [None, n_step, n_input])
        self.Y = tf.placeholder(tf.float32, [None, n_class])

        w1 = tf.Variable(tf.truncated_normal(shape=(7, 7, 1, 10), stddev=0.1))
        bias1 = tf.Variable(tf.constant(0.1, shape=[10]))

        W = tf.Variable(tf.truncated_normal(shape=(n_hidden, n_class),stddev=0.1))
        b = tf.Variable(tf.constant(0.1, shape=[1]))

        self.cell = tf.nn.rnn_cell.LSTMCell(n_hidden)

        outputs, states = tf.nn.dynamic_rnn(self.cell,self.X,dtype=tf.float32)

        outputs = tf.transpose(outputs, [1, 0, 2])
        outputs = outputs[-1]
        self.model_z = tf.add(tf.matmul(outputs, W), b)

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.model_z, labels=self.Y))

        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)
        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()


    def restoring_variables(self):
        som = [f for f in os.listdir(self.meta_json["MODEL_NAME"] + '-'  + self.meta_json["STARTED_DATESTRING"] + '/.') if f.lower().endswith('iter.ckpt')]
        soi = []
        for i in som:
            soi.append(int(i.split('.')[0][5:-4]))
        self.saver.restore(self.sess, self.meta_json["MODEL_NAME"] + '-'  + self.meta_json["STARTED_DATESTRING"] + "/model_lstm.ckpt")
        self.ccc = 100

        return

    def train(self, x_train, y_train, x_test, y_test):
        # x_train, x_test : (number of trials , number of input, number of step)
        # y_train, y_test : (number of trials , 2)

        loss_ = self.cost
        loss_hist = tf.summary.scalar('loss', loss_)
        merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(
            './' + self.meta_json["MODEL_NAME"] + self.meta_json["STARTED_DATESTRING"] + self.meta_json["LOGDIR_ROOT"],
            self.sess.graph)

        x_tt = x_train.reshape((x_train.shape[0], -1, 3) ).astype(np.float32)
        # x_tt = tf.Variable( tf.cast( tf.reshape(x_train, (x_train.shape[0], -1) ), dtype=tf.float32) )
        y_mod = np.zeros((y_train.shape[0],2), dtype=np.int16)
        y_train = y_train.astype(np.float32)
        for i in range(y_train.shape[0]):
            y_mod[i, int(((y_train[i]) + 1)/2)] = 1
        # y = tf.Variable(y_mod,dtype=tf.float32)
        y = y_mod.astype(np.float32)

        for i in range(int(self.meta_json["NUM_STEPS"])):
            self.sess.run(self.optimizer , feed_dict = {self.X: x_tt, self.Y: y})
            if i%(self.meta_json["CHECKPOINT_EVERY"]) == 0:
                self.ccc += 1
                print('[{0}/{1}] : {2} -- loss : {3}'.format(self.ccc , int(self.meta_json["NUM_STEPS"]/self.meta_json["CHECKPOINT_EVERY"]) , self.accuracy_score( x_test, y_test), self.sess.run(self.cost, feed_dict = {self.X: x_tt, self.Y: y}) ) )
            try:
                save_path = self.saver.save(self.sess, self.meta_json["MODEL_NAME"] + '-' + self.meta_json[
                    "STARTED_DATESTRING"] + "/model_lstm.ckpt")
                save_path = self.saver.save(self.sess, self.meta_json["MODEL_NAME"] + '-' + self.meta_json[
                    "STARTED_DATESTRING"] + "/model" + str(i) + "iter_lstm.ckpt")
                summary = self.sess.run(merged, feed_dict = {self.X: x_tt, self.Y: y})
                self.writer.add_summary(summary, i)

            except:
                print("checkpoint didnt saved. pleas be sure that is well saved")

        self.writer.close()

        return 0

    def accuracy_score(self, x_test, y_test):
        x_tt = x_test.reshape((x_test.shape[0], -1, 3) ).astype(np.float32)
        y_mod = np.zeros((y_test.shape[0],2), dtype=np.int16)
        y_train = y_test.astype(np.float32)
        for i in range(y_train.shape[0]):
            y_mod[i, int(((y_train[i]) + 1)/2)] = 1
        # y = tf.Variable(y_mod,dtype=tf.float32)
        y = y_mod.astype(np.float32)
        is_correct = tf.equal(tf.argmax(self.model_z,1), tf.argmax(self.Y,1) )
        accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
        self.acc = self.sess.run(accuracy, feed_dict={self.X :x_tt, self.Y:y})

        return self.acc


class CNN2D_LSTM:
    # origianl input data size : 61x61x64(frequency x time point x channel number) to 1D image
    # reference : A novel deep learning approach for classification of EEG motor imagerty signals, Yousef Razael Tabar and Ugur Halici, 2017, Journal of Neural Engineering
    #
    def __init__(self, meta_json, sub, cv):
        input_layer = 3
        self.meta_json = meta_json
        self.ccc = 0
        self.sub = sub + 1
        self.cv = cv
        self.puts_out_tw = 20
        return

    def restoring_variables(self):
        som = [f for f in os.listdir(self.meta_json["MODEL_NAME"] + '-'  + self.meta_json["STARTED_DATESTRING"] + '/.') if f.lower().endswith('iter.ckpt')]
        soi = []
        for i in som:
            soi.append(int(i.split('.')[0][5:-4]))
        self.saver.restore(self.sess, self.meta_json["MODEL_NAME"] + '-'  + self.meta_json["STARTED_DATESTRING"] + "/model.ckpt")
        self.ccc = 100

    def initialize_variables(self):

        tf.reset_default_graph()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        meta_json = self.meta_json
        learning_rate = meta_json["LEARNING_RATE"]
        total_epoch = meta_json["NUM_STEPS"]
        batch_size = meta_json["SAMPLE_SIZE"]

        # text_length = self._length(self.input_text) # size = batch_size I dont think we need this argument
        initializer = tf.contrib.layers.xavier_initializer()

        # reshape x_train to 2D image
        # For DNN with 5 channels only : E6(Fz), E10(Fp1), E5(Fp2), E18(F7), E58(F8)
        # For CNN2D with 4 channels only : E10(Fp1), E5(Fp2), E18(F7), E58(F8)
        # x_train = np.zeros((100,64,61,61))
        # shape of the image = (batch, 61*2, 61*2)

        # # controls the learning rate decay.
        # batch = tf.Variable(0)
        #
        # learning_rate = tf.train.exponential_decay(
        #     1e-4,  # Base learning rate.
        #     batch * batch_size,  # Current index into the dataset.
        #     train_size,  # Decay step.
        #     0.95,  # Decay rate.
        #     staircase=True)


        channel_index = (4, 9, 17, 57)

        # self.sess = tf.Session()
        # conv network
        # if self.meta_json["GPU"] is True:
        #     devicemapping = '/gpu:0'
        # else :
        #     devicemapping = '/cpu:0'

        # conv network
        self.x_in = tf.placeholder(shape=[None, 61 * 2, 61 * 2, 1], dtype=tf.float32)
        self.y_ = tf.placeholder(dtype=tf.float32, shape=[None, 2])
        self.behav = tf.placeholder(dtype= tf.float32, shape=[None, 2]) # behavioral input
        # with tf.device(devicemapping):

        w1 = tf.Variable(tf.truncated_normal(shape=(7, 7, 1, 10), stddev=0.1))
        bias1 = tf.Variable(tf.constant(0.1, shape=[10]))
        conv1 = tf.nn.conv2d(self.x_in, w1, strides=[1, 2, 2, 1], padding="VALID")  # (?, 116, 116, 16)
        conv1_out = tf.nn.relu(conv1)
        pool1 = tf.nn.max_pool(conv1_out, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')  # (?, 29, 29, 10)

        w2 = tf.Variable(tf.truncated_normal(shape=(7, 7, 10, 30), stddev=0.1))
        bias2 = tf.Variable(tf.constant(0.1, shape=[30]))
        conv2 = tf.nn.conv2d(pool1, w2, strides=[1, 1, 1, 1], padding="VALID")
        conv2_out = tf.nn.relu(conv2)
        self.pool2 = tf.nn.max_pool(conv2_out, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')  # (?, 12, 12, 32)

        # This model cannot be structurally deep since this model applied 1d convolution
        self.gap = tf.reduce_mean(tf.reshape(self.pool2, shape=(-1,12*12,30)), axis=1)

        self.weight_ = tf.Variable(tf.zeros((30, 2), dtype=tf.float32), dtype=tf.float32)
        self.bias_ = tf.Variable(tf.zeros((2), dtype=tf.float32), dtype=tf.float32)

        self.y = tf.matmul(self.gap, self.weight_)

        self.softmax_ = tf.exp(self.y)/tf.reduce_sum(tf.exp(self.y))

        ## TODO: implement bidirectional LSTM

        n_input = 2 # state, goal, MB/MF
        n_step = self.puts_out_tw
        n_hidden = 100
        d_a_size = 200
        r_size = 50
        fc_size = 500
        num_classes = 2
        p_coef = 1.0


        self.Z = tf.concat([self.behav, self.softmax_],axis=1)
        Z_ = tf.concat([self.Z, tf.zeros((model1.puts_out_tw - 1, self.Z.shape[1]))], axis=0)
        X_tt = []
        for xtpi in range(specs.shape[0]):
            X_tt.append(Z_[xtpi:xtpi + model1.puts_out_tw, :])
        xtt = tf.Variable(X_tt)

        self.X = xtt
        # Bidirectional(Left&Right) Recurrent Structure
        with tf.name_scope("bi-lstm"):
            fw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
            bw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
            (self.output_fw, self.output_bw), states = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell,
                                                                                       cell_bw=bw_cell,
                                                                                       inputs=self.X,
                                                                                       dtype=tf.float32)
            self.H = tf.concat([self.output_fw, self.output_bw], axis=2)
            H_reshape = tf.reshape(self.H, [-1, 2 * n_hidden])

        with tf.name_scope("self-attention"):
            self.W_s1 = tf.get_variable("W_s1", shape=[2*n_hidden, d_a_size], initializer=initializer)
            _H_s1 = tf.nn.tanh(tf.matmul(H_reshape, self.W_s1))
            self.W_s2 = tf.get_variable("W_s2", shape=[d_a_size, r_size], initializer=initializer)
            _H_s2 = tf.matmul(_H_s1, self.W_s2)
            _H_s2_reshape = tf.transpose(tf.reshape(_H_s2, [-1, self.puts_out_tw, r_size]), [0, 2, 1])
            self.A = tf.nn.softmax(_H_s2_reshape, name="attention")

        with tf.name_scope("sentence-embedding"):
            self.M = tf.matmul(self.A, self.H)

        with tf.name_scope("fully-connected"):
            # self.M_pool = tf.reduce_mean(self.M, axis=1)
            # W_fc = tf.get_variable("W_fc", shape=[2 * hidden_size, fc_size], initializer=initializer)
            self.M_flat = tf.reshape(self.M, shape=[-1, 2 * n_hidden * r_size])
            W_fc = tf.get_variable("W_fc", shape=[2 * n_hidden * r_size, fc_size], initializer=initializer)
            b_fc = tf.Variable(tf.constant(0.1, shape=[fc_size]), name="b_fc")
            self.fc = tf.nn.relu(tf.nn.xw_plus_b(self.M_flat, W_fc, b_fc), name="fc")

        with tf.name_scope("output"):
            W_output = tf.get_variable("W_output", shape=[fc_size, num_classes], initializer=initializer)
            b_output = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b_output")
            self.logits = tf.nn.xw_plus_b(self.fc, W_output, b_output, name="logits")
            self.predictions = tf.argmax(self.logits, 1, name="predictions")

        with tf.name_scope("penalization"):
            self.AA_T = tf.matmul(self.A, tf.transpose(self.A, [0, 2, 1]))
            self.I = tf.reshape(tf.tile(tf.eye(r_size), [tf.shape(self.A)[0], 1]), [-1, r_size, r_size])
            self.P = tf.square(tf.norm(self.AA_T - self.I, axis=[-2, -1], ord="fro"))

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y)
            self.loss_P = tf.reduce_mean(self.P * p_coef)
            self.loss = tf.reduce_mean(losses) + self.loss_P

        with tf.name_scope("optimizer"):
            self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.Y, axis=1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name="accuracy")
        self.sess.run(tf.global_variables_initializer())

        # self.train_step = tf.train.AdamOptimizer(learning_rate=self.meta_json["LEARNING_RATE"]).minimize(self.loss)
        # self.sess.run(tf.global_variables_initializer())


        self.saver = tf.train.Saver()

        self.writer = tf.summary.FileWriter(
            './' + self.meta_json["MODEL_NAME"] + self.meta_json["STARTED_DATESTRING"] + self.meta_json["LOGDIR_ROOT"],
            self.sess.graph)

        return

    def train(self, x_train, y_train, behav_train, x_test, y_test, behav_test):

        if self.meta_json["rest"] is True:
            self.initialize_variables()
            self.restoring_variables()
        else:
            self.initialize_variables()

        # x_train and y_train
        channel_index = (4, 9, 17, 57)
        temp1 =np.concatenate((x_train[:,4,:,:],x_train[:,9,:,:]), axis=1)
        temp2 =np.concatenate((x_train[:,17,:,:],x_train[:,57,:,:]), axis=1)
        x_tt = np.concatenate((temp1,temp2),axis=2).reshape((x_train.shape[0],61*2,61*2,1))

        y_mod = np.zeros((y_train.shape[0],2), dtype=np.int16)
        y_train = y_train.astype(np.float32)
        for i in range(y_train.shape[0]):
            y_mod[i, int(((y_train[i]) + 1)/2)] = 1
        # y = tf.Variable(y_mod,dtype=tf.float32)
        y = y_mod.astype(np.float32)


        loss_ = self.loss
        loss_hist = tf.summary.scalar('loss', loss_)
        merged = tf.summary.merge_all()
        # x_tt = tf.Variable( tf.cast( tf.reshape(x_train, (x_train.shape[0], -1) ), dtype=tf.float32) )
        y_mod = np.zeros((y_train.shape[0],2), dtype=np.int16)
        y_train = y_train.astype(np.float32)
        for i in range(y_train.shape[0]):
            y_mod[i, int(((y_train[i]) + 1)/2)] = 1
        # y = tf.Variable(y_mod,dtype=tf.float32)
        y = y_mod.astype(np.float32)
        for i in range(int(self.meta_json["NUM_STEPS"])):
            totbat = x_tt.shape[0]
            fortemp = 1
            while (totbat/fortemp) > 20:
                fortemp += 1
            e_bat =int(totbat / fortemp)
            for j in range(fortemp):
                self.sess.run(self.optimizer, feed_dict={self.x_in: x_tt[e_bat * j:e_bat * (j + 1), :, :, :],
                                                          self.y_: y[e_bat * j:e_bat * (j + 1), :],
                                                          self.behav: behav_train
                                                          })
            if i%(1000) == 0:
                self.ccc += 1
                print('[{0}/{1}] : {2} -- loss : {3}'.format(self.ccc , int(self.meta_json["NUM_STEPS"]/1000) , self.accuracy_score(x_test, y_test, behav_test), self.sess.run(self.loss, feed_dict = {self.x_in: x_tt, self.y_: y,
                                                          self.behav: behav_train}) ) )
                try:
                    self.CAMgen(x_test,y_test)
                    save_path = self.saver.save(self.sess, self.meta_json["MODEL_NAME"] + '-' + self.meta_json[
                        "STARTED_DATESTRING"] + "/model.ckpt")
                    save_path = self.saver.save(self.sess, self.meta_json["MODEL_NAME"] + '-' + self.meta_json[
                        "STARTED_DATESTRING"] + "/model" + str(i) + "iter.ckpt")
                    summary = self.sess.run(merged, feed_dict={self.x_in: x_tt, self.y_: y})
                    self.writer.add_summary(summary, i)
                except:
                    print("checkpoint didnt saved. pleas be sure that is well saved")

        #
        # for i in range(int(self.meta_json["NUM_STEPS"])):
        #     totbat = x_train.shape[0]
        #     fortemp = 1
        #     while (totbat/fortemp) > 20:
        #         fortemp += 1
        #     e_bat =int(totbat / fortemp)
        #     for j in range(fortemp):
        #         self.sess.run(self.train_step, feed_dict = {self.x_in: x_train[e_bat*j:e_bat*(j+1),:,:,:,:], self.y_: y[e_bat*j:e_bat*(j+1),:]})
        #     if i%(1000) == 0:
        #         self.ccc += 1
        #         totbat_ = x_test.shape[0]
        #         fortemp_ = 1
        #         while (totbat_ / fortemp_) > 20:
        #             fortemp_ += 1
        #         e_bat_ = int(totbat_ / fortemp_)
        #         dap = []
        #         for k in range(fortemp_):
        #             dap.append(self.accuracy_score(x_test[e_bat*k:e_bat*(k+1),:,:,:], y_test[e_bat*k:e_bat*(k+1)]))
        #         ls =  0
        #         for j in range(fortemp):
        #             ls += self.sess.run(self.loss, feed_dict={self.x_in: x_train[e_bat*j:e_bat*(j+1),:,:,:,:], self.y_: y[e_bat*j:e_bat*(j+1),:]})
        #
        #
        #         print('[{0}/{1}] : {2} -- loss : {3}'.format(self.ccc , int(self.meta_json["NUM_STEPS"]/1000) , np.mean(dap), ls) )
        #         try:
        #             self.CAMgen(x_test,y_test)
        #             save_path = self.saver.save(self.sess, self.meta_json["MODEL_NAME"] + '-' + self.meta_json[
        #                 "STARTED_DATESTRING"] + "/model.ckpt")
        #             save_path = self.saver.save(self.sess, self.meta_json["MODEL_NAME"] + '-' + self.meta_json[
        #                 "STARTED_DATESTRING"] + "/model" + str(i) + "iter.ckpt")
        #             summary = self.sess.run(merged, feed_dict={self.x_in: x_train[e_bat*j:e_bat*(j+1),:,:,:,:], self.y_: y[e_bat*j:e_bat*(j+1),:]})
        #             self.writer.add_summary(summary, i)
        #         except:
        #             print("checkpoint didnt saved. pleas be sure that is well saved")
        self.writer.close()

        return 0


    def accuracy_score(self, x_test, y_test,behav_test):
        channel_index = (4, 9, 17, 57)
        temp1 =np.concatenate((x_test[:,4,:,:],x_test[:,9,:,:]), axis=1)
        temp2 =np.concatenate((x_test[:,17,:,:],x_test[:,57,:,:]), axis=1)
        x_tt = np.concatenate((temp1,temp2),axis=2).reshape((x_test.shape[0],61*2,61*2,1))

        y_mod = np.zeros((y_test.shape[0],2), dtype=np.int16)
        y_test = y_test.astype(np.float32)
        for i in range(y_test.shape[0]):
            y_mod[i, int(((y_test[i]) + 1)/2)] = 1
        # y = tf.Variable(y_mod,dtype=tf.float32)
        y = y_mod.astype(np.float32)


        channel_index = (4, 9, 17, 57)
        temp1 =np.concatenate((x_test[:,4,:,:],x_test[:,9,:,:]), axis=1)
        temp2 =np.concatenate((x_test[:,17,:,:],x_test[:,57,:,:]), axis=1)
        x_tt = np.concatenate((temp1,temp2),axis=2).reshape((x_test.shape[0],61*2,61*2,1))

        # y = tf.Variable(y_mod,dtype=tf.float32)
        y__ = y_mod.astype(np.float32)
        y_pred = np.argmax(self.sess.run(self.y, feed_dict={self.x_in: x_tt, self.behav: behav_test}),axis=1)
        self.acc = np.average(np.equal(y_pred, np.argmax(y__, axis = 1) ))

        return self.acc

class CNN_LSTM:
    # origianl input data size : 61x61x64(frequency x time point x channel number) to 1D image
    # reference : A novel deep learning approach for classification of EEG motor imagerty signals, Yousef Razael Tabar and Ugur Halici, 2017, Journal of Neural Engineering
    #
    def __init__(self, meta_json):
        input_layer = 3
        self.meta_json = meta_json
        self.meta_json['addto'] = self.meta_json['MODEL_NAME']
        self.build_addon()
        self.ccc = 0

        return

    def build_addon(self):
        if self.meta_json['addto'] == 'nothing':
            print('BOOM in half')
        else:
            self.model = eval(self.meta_json['addto'] + '(self.meta_json)')
            self.model.initialize_variables()
            self.model.restoring_variables()
        return

    def puts_out(self, spectro_train, spectro_test): # returning last hidden layer right before the dense layer.
        if self.meta_json['addto'] == 'CNN3D_TFC_CAM':
            ch = np.array([63,62,64,17,10,5,1,61,8,11,2,18,13,9,6,3,59,58,12,60,19,14,57,56,23,15,7,54,53,55])
            channel_index = tuple((ch-1).tolist())
            x_train = spectro_train.reshape((spectro_train.shape[0],spectro_train.shape[1],spectro_train.shape[2],spectro_train.shape[3],1))
            x_trains = x_train[:,channel_index, :, :30, :]
            x_train = np.zeros((x_train.shape[0], 30, 30, x_trains.shape[3], 1)) # ? 64 30 30 1
            for tr in range(x_train.shape[2]):
                x_train[:,:,tr,:,:] = (x_trains[:,:,2*tr,:,:] +x_trains[:,:,2*tr+1,:,:]) /2

            x_test = spectro_test.reshape((spectro_test.shape[0], spectro_test.shape[1], spectro_test.shape[2], spectro_test.shape[3], 1))
            x_trains = x_test[:, channel_index, :, :30, :]
            x_test = np.zeros((x_test.shape[0], 30, 30, x_trains.shape[3], 1))  # ? 64 30 30 1
            for tr in range(x_test.shape[2]):
                x_test[:, :, tr, :, :] = (x_trains[:, :, 2 * tr, :, :] + x_trains[:, :, 2 * tr + 1, :, :]) / 2

            y_train = self.model.sess.run(self.model.pool2,feed_dict={self.model.x_in : x_train})
            y_test = self.model.sess.run(self.model.pool2,feed_dict={self.model.x_in : x_test})


            return (y_train, y_test)
        elif self.meta_json['addto'] == 'CNN2DwithCAM':
            # self.model.saver.restore(self.model.sess, self.meta_json["MODEL_NAME"] + '-' + self.meta_json[
            #     "STARTED_DATESTRING"] + "/model.ckpt")
            temp1 = np.concatenate((spectro_train[:, 4, :, :], spectro_train[:, 9, :, :]), axis=1)
            temp2 = np.concatenate((spectro_train[:, 17, :, :], spectro_train[:, 57, :, :]), axis=1)
            x_train = np.concatenate((temp1, temp2), axis=2).reshape((spectro_train.shape[0], 61 * 2, 61 * 2, 1))
            temp1 = np.concatenate((spectro_test[:, 4, :, :], spectro_test[:, 9, :, :]), axis=1)
            temp2 = np.concatenate((spectro_test[:, 17, :, :], spectro_test[:, 57, :, :]), axis=1)
            x_test = np.concatenate((temp1, temp2), axis=2).reshape((spectro_test.shape[0], 61 * 2, 61 * 2, 1))

            y_train = self.model.sess.run(self.model.pool2,feed_dict={self.model.x_in : x_train})
            y_test = self.model.sess.run(self.model.pool2,feed_dict={self.model.x_in : x_test})


            return (y_train, y_test)
        else:
            y_train = np.zeros((spectro_train.shape[0],))
            y_test = np.zeros((spectro_test.shape[0],))

            return (y_train, y_test)

    def init_LSTM(self):

        if self.meta_json['addto'] == 'CNN2DwithCAM':
            tf.reset_default_graph()
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            meta_json = self.meta_json
            learning_rate = meta_json["LEARNING_RATE"]
            total_epoch = meta_json["NUM_STEPS"]
            batch_size = meta_json["SAMPLE_SIZE"]


            n_input = 12*30  # state, goal, MB/MF
            n_step = 12
            n_hidden = 128
            n_class = 2

            self.X = tf.placeholder(tf.float32, [None, n_step, n_input])
            self.Y = tf.placeholder(tf.float32, [None, n_class])


            W = tf.Variable(tf.truncated_normal(shape=(n_hidden, n_class), stddev=0.1))
            b = tf.Variable(tf.constant(0.1, shape=[1]))

            self.cell = tf.nn.rnn_cell.LSTMCell(n_hidden)

            outputs, states = tf.nn.dynamic_rnn(self.cell, self.X, dtype=tf.float32)

            outputs = tf.transpose(outputs, [1, 0, 2])
            outputs = outputs[-1]
            self.model_z = tf.add(tf.matmul(outputs, W), b)

            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.model_z, labels=self.Y))

            self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)
            self.sess.run(tf.global_variables_initializer())

            self.saver = tf.train.Saver()

        elif self.meta_json['addto'] == 'CNN3D_TFC_CAM':
            tf.reset_default_graph()
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            meta_json = self.meta_json
            learning_rate = meta_json["LEARNING_RATE"]
            total_epoch = meta_json["NUM_STEPS"]
            batch_size = meta_json["SAMPLE_SIZE"]


            n_input = 9*9*30  # state, goal, MB/MF
            n_step = 9
            n_hidden = 128
            n_class = 2

            self.X = tf.placeholder(tf.float32, [None, n_step, n_input])
            self.Y = tf.placeholder(tf.float32, [None, n_class])

            W = tf.Variable(tf.truncated_normal(shape=(n_hidden, n_class), stddev=0.1))
            b = tf.Variable(tf.constant(0.1, shape=[1]))

            self.cell = tf.nn.rnn_cell.LSTMCell(n_hidden)

            outputs, states = tf.nn.dynamic_rnn(self.cell, self.X, dtype=tf.float32)

            outputs = tf.transpose(outputs, [1, 0, 2])
            outputs = outputs[-1]
            self.model_z = tf.add(tf.matmul(outputs, W), b)

            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.model_z, labels=self.Y))

            self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)
            self.sess.run(tf.global_variables_initializer())

            self.saver = tf.train.Saver()



    def restoring_variables(self):
        som = [f for f in os.listdir(self.meta_json["MODEL_NAME"] + '-'  + self.meta_json["STARTED_DATESTRING"] + '/.') if f.lower().endswith('iter.ckpt')]
        soi = []
        for i in som:
            soi.append(int(i.split('.')[0][5:-4]))
        self.saver.restore(self.sess, self.meta_json["MODEL_NAME"] + '-'  + self.meta_json["STARTED_DATESTRING"] + "/model_lstm.ckpt")
        self.ccc = 100

        return

    def train(self, x_train, y_train, x_test, y_test):
        # x_train, x_test : (number of trials , number of input, number of step)
        # y_train, y_test : (number of trials , 2)

        loss_ = self.cost
        loss_hist = tf.summary.scalar('loss', loss_)
        merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(
            './' + self.meta_json["MODEL_NAME"] + self.meta_json["STARTED_DATESTRING"] + self.meta_json["LOGDIR_ROOT"],
            self.sess.graph)

        x_tt = x_train.reshape((x_train.shape[0],x_train.shape[1], x_train.shape[2]) ).astype(np.float32)
        # x_tt = tf.Variable( tf.cast( tf.reshape(x_train, (x_train.shape[0], -1) ), dtype=tf.float32) )
        y_mod = np.zeros((y_train.shape[0],2), dtype=np.int16)
        y_train = y_train.astype(np.float32)
        for i in range(y_train.shape[0]):
            y_mod[i, int(((y_train[i]) + 1)/2)] = 1
        # y = tf.Variable(y_mod,dtype=tf.float32)
        y = y_mod.astype(np.float32)

        for i in range(int(self.meta_json["NUM_STEPS"])):

            totbat = x_train.shape[0]
            fortemp = 100
            while (totbat/fortemp) > 20:
                fortemp += 1
            e_bat =int(totbat / fortemp)
            for j in range(fortemp):
                self.sess.run(self.optimizer , feed_dict = {self.X: x_tt[e_bat*j:e_bat*(j+1),:,:], self.Y: y[e_bat*j:e_bat*(j+1),:]})
            if i%(self.meta_json["CHECKPOINT_EVERY"]) == 0:
                self.ccc += 1
                print('[{0}/{1}] : {2} -- loss : {3}'.format(self.ccc , int(self.meta_json["NUM_STEPS"]/self.meta_json["CHECKPOINT_EVERY"]) , self.accuracy_score( x_test, y_test), self.sess.run(self.cost, feed_dict = {self.X: x_tt, self.Y: y}) ) )
            try:
                save_path = self.saver.save(self.sess, self.meta_json["MODEL_NAME"] + '-' + self.meta_json[
                    "STARTED_DATESTRING"] + "/model_lstm.ckpt")
                save_path = self.saver.save(self.sess, self.meta_json["MODEL_NAME"] + '-' + self.meta_json[
                    "STARTED_DATESTRING"] + "/model" + str(i) + "iter_lstm.ckpt")
                summary = self.sess.run(merged, feed_dict = {self.X: x_tt, self.Y: y})
                self.writer.add_summary(summary, i)

            except:
                print("checkpoint didnt saved. pleas be sure that is well saved")

        self.writer.close()

        return 0

    def accuracy_score(self, x_test, y_test):
        x_tt = x_test.reshape((x_test.shape[0], x_test.shape[1], x_test.shape[2]) ).astype(np.float32)
        y_mod = np.zeros((y_test.shape[0],2), dtype=np.int16)
        y_train = y_test.astype(np.float32)
        for i in range(y_train.shape[0]):
            y_mod[i, int(((y_train[i]) + 1)/2)] = 1
        # y = tf.Variable(y_mod,dtype=tf.float32)
        y = y_mod.astype(np.float32)
        is_correct = tf.equal(tf.argmax(self.model_z,1), tf.argmax(self.Y,1) )
        accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
        self.acc = self.sess.run(accuracy, feed_dict={self.X :x_tt, self.Y:y})

        return self.acc


class CNN_LSTM_GAP:
    # origianl input data size : 61x61x64(frequency x time point x channel number) to 1D image
    # reference : A novel deep learning approach for classification of EEG motor imagerty signals, Yousef Razael Tabar and Ugur Halici, 2017, Journal of Neural Engineering
    #
    def __init__(self, meta_json):
        input_layer = 3
        self.meta_json = meta_json
        self.meta_json['addto'] = self.meta_json['MODEL_NAME']
        self.build_addon()
        self.ccc = 0

        return

    def build_addon(self):
        if self.meta_json['addto'] == 'nothing':
            print('BOOM in half')
        else:
            self.model = eval(self.meta_json['addto'] + '(self.meta_json)')
            self.model.initialize_variables()
            self.model.restoring_variables()
        return

    def puts_out(self, spectro_train, spectro_test): # returning last hidden layer right before the dense layer.
        if self.meta_json['addto'] == 'CNN3D_TFC_CAM':
            ch = np.array([63,62,64,17,10,5,1,61,8,11,2,18,13,9,6,3,59,58,12,60,19,14,57,56,23,15,7,54,53,55])
            channel_index = tuple((ch-1).tolist())
            x_train = spectro_train.reshape((spectro_train.shape[0],spectro_train.shape[1],spectro_train.shape[2],spectro_train.shape[3],1))
            x_trains = x_train[:,channel_index, :, :30, :]
            x_train = np.zeros((x_train.shape[0], 30, 30, x_trains.shape[3], 1)) # ? 64 30 30 1
            for tr in range(x_train.shape[2]):
                x_train[:,:,tr,:,:] = (x_trains[:,:,2*tr,:,:] +x_trains[:,:,2*tr+1,:,:]) /2

            x_test = spectro_test.reshape((spectro_test.shape[0], spectro_test.shape[1], spectro_test.shape[2], spectro_test.shape[3], 1))
            x_trains = x_test[:, channel_index, :, :30, :]
            x_test = np.zeros((x_test.shape[0], 30, 30, x_trains.shape[3], 1))  # ? 64 30 30 1
            for tr in range(x_test.shape[2]):
                x_test[:, :, tr, :, :] = (x_trains[:, :, 2 * tr, :, :] + x_trains[:, :, 2 * tr + 1, :, :]) / 2

            y_train = self.model.sess.run(self.model.gap,feed_dict={self.model.x_in : x_train})
            y_test = self.model.sess.run(self.model.gap,feed_dict={self.model.x_in : x_test})


            return (y_train, y_test)
        elif self.meta_json['addto'] == 'CNN2DwithCAM':
            # self.model.saver.restore(self.model.sess, self.meta_json["MODEL_NAME"] + '-' + self.meta_json[
            #     "STARTED_DATESTRING"] + "/model.ckpt")
            temp1 = np.concatenate((spectro_train[:, 4, :, :], spectro_train[:, 9, :, :]), axis=1)
            temp2 = np.concatenate((spectro_train[:, 17, :, :], spectro_train[:, 57, :, :]), axis=1)
            x_train = np.concatenate((temp1, temp2), axis=2).reshape((spectro_train.shape[0], 61 * 2, 61 * 2, 1))
            temp1 = np.concatenate((spectro_test[:, 4, :, :], spectro_test[:, 9, :, :]), axis=1)
            temp2 = np.concatenate((spectro_test[:, 17, :, :], spectro_test[:, 57, :, :]), axis=1)
            x_test = np.concatenate((temp1, temp2), axis=2).reshape((spectro_test.shape[0], 61 * 2, 61 * 2, 1))

            y_train = self.model.sess.run(self.model.gap,feed_dict={self.model.x_in : x_train})
            y_test = self.model.sess.run(self.model.gap,feed_dict={self.model.x_in : x_test})


            return (y_train, y_test)
        else:
            y_train = np.zeros((spectro_train.shape[0],))
            y_test = np.zeros((spectro_test.shape[0],))

            return (y_train, y_test)

    def init_LSTM(self):
        tf.reset_default_graph()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        meta_json = self.meta_json
        learning_rate = meta_json["LEARNING_RATE"]
        total_epoch = meta_json["NUM_STEPS"]
        batch_size = meta_json["SAMPLE_SIZE"]


        n_input = 30  # state, goal, MB/MF
        n_step = 1
        n_hidden = 128
        n_class = 2

        self.X = tf.placeholder(tf.float32, [None, n_step, n_input])
        self.Y = tf.placeholder(tf.float32, [None, n_class])


        W = tf.Variable(tf.truncated_normal(shape=(n_hidden, n_class), stddev=0.1))
        b = tf.Variable(tf.constant(0.1, shape=[1]))

        self.cell = tf.nn.rnn_cell.LSTMCell(n_hidden)

        outputs, states = tf.nn.dynamic_rnn(self.cell, self.X, dtype=tf.float32)

        outputs = tf.transpose(outputs, [1, 0, 2])
        outputs = outputs[-1]
        self.model_z = tf.add(tf.matmul(outputs, W), b)

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.model_z, labels=self.Y))

        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)
        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()



    def restoring_variables(self):
        som = [f for f in os.listdir(self.meta_json["MODEL_NAME"] + '-'  + self.meta_json["STARTED_DATESTRING"] + '/.') if f.lower().endswith('iter.ckpt')]
        soi = []
        for i in som:
            soi.append(int(i.split('.')[0][5:-4]))
        self.saver.restore(self.sess, self.meta_json["MODEL_NAME"] + '-'  + self.meta_json["STARTED_DATESTRING"] + "/model_lstm.ckpt")
        self.ccc = 100

        return

    def train(self, x_train, y_train, x_test, y_test):
        # x_train, x_test : (number of trials , number of input, number of step)
        # y_train, y_test : (number of trials , 2)

        loss_ = self.cost
        loss_hist = tf.summary.scalar('loss', loss_)
        merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(
            './' + self.meta_json["MODEL_NAME"] + self.meta_json["STARTED_DATESTRING"] + self.meta_json["LOGDIR_ROOT"],
            self.sess.graph)

        x_tt = x_train.reshape((x_train.shape[0],x_train.shape[1], x_train.shape[2]) ).astype(np.float32)
        # x_tt = tf.Variable( tf.cast( tf.reshape(x_train, (x_train.shape[0], -1) ), dtype=tf.float32) )
        y_mod = np.zeros((y_train.shape[0],2), dtype=np.int16)
        y_train = y_train.astype(np.float32)
        for i in range(y_train.shape[0]):
            y_mod[i, int(((y_train[i]) + 1)/2)] = 1
        # y = tf.Variable(y_mod,dtype=tf.float32)
        y = y_mod.astype(np.float32)

        for i in range(int(self.meta_json["NUM_STEPS"])):

            totbat = x_train.shape[0]
            fortemp = 100
            while (totbat/fortemp) > 20:
                fortemp += 1
            e_bat =int(totbat / fortemp)
            for j in range(fortemp):
                self.sess.run(self.optimizer , feed_dict = {self.X: x_tt[e_bat*j:e_bat*(j+1),:,:], self.Y: y[e_bat*j:e_bat*(j+1),:]})
            if i%(self.meta_json["CHECKPOINT_EVERY"]) == 0:
                self.ccc += 1
                print('[{0}/{1}] : {2} -- loss : {3}'.format(self.ccc , int(self.meta_json["NUM_STEPS"]/self.meta_json["CHECKPOINT_EVERY"]) , self.accuracy_score( x_test, y_test), self.sess.run(self.cost, feed_dict = {self.X: x_tt, self.Y: y}) ) )
            try:
                save_path = self.saver.save(self.sess, self.meta_json["MODEL_NAME"] + '-' + self.meta_json[
                    "STARTED_DATESTRING"] + "/model_lstm.ckpt")
                save_path = self.saver.save(self.sess, self.meta_json["MODEL_NAME"] + '-' + self.meta_json[
                    "STARTED_DATESTRING"] + "/model" + str(i) + "iter_lstm.ckpt")
                summary = self.sess.run(merged, feed_dict = {self.X: x_tt, self.Y: y})
                self.writer.add_summary(summary, i)

            except:
                print("checkpoint didnt saved. pleas be sure that is well saved")

        self.writer.close()

        return 0

    def accuracy_score(self, x_test, y_test):
        x_tt = x_test.reshape((x_test.shape[0], x_test.shape[1], x_test.shape[2]) ).astype(np.float32)
        y_mod = np.zeros((y_test.shape[0],2), dtype=np.int16)
        y_train = y_test.astype(np.float32)
        for i in range(y_train.shape[0]):
            y_mod[i, int(((y_train[i]) + 1)/2)] = 1
        # y = tf.Variable(y_mod,dtype=tf.float32)
        y = y_mod.astype(np.float32)
        is_correct = tf.equal(tf.argmax(self.model_z,1), tf.argmax(self.Y,1) )
        accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
        self.acc = self.sess.run(accuracy, feed_dict={self.X :x_tt, self.Y:y})

        return self.acc



if __name__ == '__main__':
    meta_json=dict()
    meta_json["NUM_STEPS"] = 1e4
    meta_json["LEARNING_RATE"] = 5e-2
    meta_json["rest"] = False
    meta_json["MODEL_NAME"]="CNN2DwithCAM"
    meta_json["STARTED_DATESTRING"]="2019-04-03"
    meta_json["LOGDIR_ROOT"]="LOG"
    meta_json["GPU"] = False
    save_dir = meta_json["MODEL_NAME"] + '-' + meta_json["STARTED_DATESTRING"] + '-CNNLSTM'
    model1 = addon_bidi_LSTM(meta_json, save_dir, 28, 1)


    meta_json["MODEL_NAME"]="CNN3D_TFC_CAM"
    meta_json["STARTED_DATESTRING"]="2019-04-03"
    meta_json["LOGDIR_ROOT"]="LOG"
    meta_json["GPU"] = False
    save_dir = meta_json["MODEL_NAME"] + '-' + meta_json["STARTED_DATESTRING"] + '-CNNLSTM'
    model2 = addon_bidi_LSTM(meta_json, save_dir, 28, 1)


    x_train = np.concatenate( (np.ones((10,64,61,61)),  np.ones((10,64,61,61)) *(-2)), axis=0)

    y_train = np.concatenate( (np.ones(10), np.ones(10)*(-1)) ,axis = 0)
    y_mod = np.zeros((y_train.shape[0], 2), dtype=np.int16)
    y_train = y_train.astype(np.float32)
    for i in range(y_train.shape[0]):
        y_mod[i, int(((y_train[i]) + 1) / 2)] = 1
"""
