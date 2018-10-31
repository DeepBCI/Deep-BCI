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

class SVM:
    # need to be changed [2018.02.05]
    def __init__(self, meta_json):
        self.model = svm.SVC()

    def train(self,x_train,y_train):
        return self.model.fit(x_train,y_train)

    def predict(self,x_test):
        estimation = self.model.predict(x_test)
        return estimation

    def accuracy_score(self, x_test, y_test):
        estimation = self.model.predict(x_test)
        return acc_func(estimation,y_test)

class DNN1Dscreen : # 1D-DNN : using flattened feature
# This model should be run by the SCREENING = false. also, in a channel by channel manner. (CHAN_WISE = true)
    def __init__(self, meta_json):
        # self.w1 = tf.Variable(tf.zeros([61*61, 900]),dtype=tf.float32)
        # self.b1 = tf.Variable(tf.zeros([900]),dtype=tf.float32)
        # self.w2 = tf.Variable(tf.zeros(([900, 500])),dtype=tf.float32)
        # self.b2 = tf.Variable(tf.zeros([500]),dtype=tf.float32)
        # self.w3 = tf.Variable(tf.zeros(([500, 200])),dtype=tf.float32)
        # self.b3 = tf.Variable(tf.zeros([200]),dtype=tf.float32)
        # self.w4 = tf.Variable(tf.zeros(([200, 100])),dtype=tf.float32)
        # self.b4 = tf.Variable(tf.zeros([100]),dtype=tf.float32)
        # self.w5 = tf.Variable(tf.zeros(([100, 40])),dtype=tf.float32)
        # self.b5 = tf.Variable(tf.zeros([40]),dtype=tf.float32)
        # self.w6 = tf.Variable(tf.zeros(([40, 15])),dtype=tf.float32)
        # self.b6 = tf.Variable(tf.zeros([15]),dtype=tf.float32)
        # self.w7 = tf.Variable(tf.zeros(([15, 2])),dtype=tf.float32)
        # self.b7 = tf.Variable(tf.zeros([2]),dtype=tf.float32)
        self.meta_json = meta_json

        self.keep_prob = 0.5

        return

    def initialize_variables(self):
        self.weight_ = []
        self.bias_ = []
        layer_pr = [61*61, 100, 15, 2]
        layer_pr = [61*61, 100, 25, 2]
        for i in range(len(layer_pr)-1):
            self.weight_.append(tf.Variable(tf.zeros([layer_pr[i], layer_pr[i+1]]),dtype=tf.float32))
            self.bias_.append(tf.Variable(tf.zeros([layer_pr[i+1]]),dtype=tf.float32))


        self.x = tf.placeholder(dtype=tf.float32, shape = [None, 61*61])
        self.y_ = tf.placeholder(dtype=tf.float32, shape = [None, 2])

        self.outlayer = []
        self.outlayer.append(self.x)
        self.keep_prob_ = tf.placeholder(tf.float32)
        for i in range(len(layer_pr)-2):
            temp_2 = tf.nn.relu(tf.matmul(self.outlayer[-1], self.weight_[i]) + self.bias_[i])
            temp_ = tf.nn.dropout (temp_2, self.keep_prob_)
            self.outlayer.append(temp_)

        # predict
        self.y = tf.matmul(self.outlayer[-1], self.weight_[len(layer_pr)-2]) + self.bias_[len(layer_pr)-2]

        # self.cross_entropy = -tf.reduce_sum(self.y_ * tf.log(self.y))

        # cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.y, labels=self.y_))
        # self.train_step = tf.train.GradientDescentOptimizer
        self.train_step = tf.train.AdamOptimizer(learning_rate=self.meta_json["LEARNING_RATE"]).minimize(self.loss)

        # self.train_step = tf.train.GradientDescentOptimizer
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        # prediction
        self.ccc = 0
        self.saver = tf.train.Saver()

        return

    def restoring_variables(self):

        som = [f for f in os.listdir(self.meta_json["MODEL_NAME"] + '-'  + self.meta_json["STARTED_DATESTRING"] + '/.') if f.lower().endswith('iter.ckpt.meta')]
        soi = []
        for i in som:
            soi.append(int(i.split('.')[0][5:-4]))
        self.saver.restore(self.sess, self.meta_json["MODEL_NAME"] + '-'  + self.meta_json["STARTED_DATESTRING"] + "/model.ckpt")
        self.ccc = max(soi)
        self.sess.run(tf.global_variables_initializer())

        return


    def train(self, x_train, y_train):
        if self.meta_json["rest"] is True:
            self.restoring_variables()
        else:
            self.initialize_variables()



        x_tt = x_train.reshape((x_train.shape[0], -1) ).astype(np.float32)
        # x_tt = tf.Variable( tf.cast( tf.reshape(x_train, (x_train.shape[0], -1) ), dtype=tf.float32) )
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

        for i in range(int(self.meta_json["NUM_STEPS"])):
            self.sess.run(self.train_step, feed_dict = {self.x: x_tt, self.y_: y, self.keep_prob_:self.keep_prob})
            if i%(1000) == 0:
                self.ccc += 1
                print('[{0}/{1}] : {2} -- loss : {3}'.format(self.ccc , int(self.meta_json["NUM_STEPS"]/1000) , self.accuracy_score( x_tt, y), self.sess.run(self.loss, feed_dict = {self.x: x_tt, self.y_: y, self.keep_prob_: 1}) ) )
                try:
                    save_path = self.saver.save(self.sess, self.meta_json["MODEL_NAME"] + '-' + self.meta_json[
                        "STARTED_DATESTRING"] + "/model.ckpt")
                    save_path = self.saver.save(self.sess, self.meta_json["MODEL_NAME"] + '-' + self.meta_json[
                        "STARTED_DATESTRING"] + "/model" + str(i) + "iter.ckpt")
                    summary = self.sess.run(merged, feed_dict = {self.x: x_tt, self.y_: y, self.keep_prob_: 1})
                    self.writer.add_summary(summary, i)

                except:
                    print("checkpoint didnt saved. pleas be sure that is well saved")

        self.writer.close()

        return 0

    # def predict(self, x_test):
    #
    #     return

    def accuracy_score(self, x_test, y_test):
        # need ed to be fixed
        # x_tt = tf.Variable(tf.reshape(x_test, (x_test.shape[0], -1)), dtype=tf.float32)
        x_tt = x_test.reshape((x_test.shape[0], -1)).astype(np.float32)
        # self.out_softmax = tf.nn.softmax()
        y_pred = np.argmax(self.sess.run(self.y, feed_dict={self.x: x_tt, self.keep_prob_: 1 }),axis=1)
        self.acc = np.average(np.equal(y_pred, np.argmax(y_test) ))

        return self.acc

class DNN1D(DNN1Dscreen):
    # For this case, use DNN with 5 channels only : E6(Fz), E10(Fp1), E5(Fp2), E18(F7), E58(F8)
    def initialize_variables(self):
        channel_index = (4,5,9,17,57)
        self.weight_ = []
        self.bias_ = []
        layer_pr = [61*61*5, 61*61, 100, 15, 2]
        layer_pr = [61*61*5, 61*61, 1500, 800, 200, 64, 15, 2]
        layer_pr = [61*61*5, 61*61, 1024, 2]






        self.x = tf.placeholder(dtype=tf.float32, shape = [None, 61*61*5])
        self.y_ = tf.placeholder(dtype=tf.float32, shape = [None, 2])
        self.keep_prob = tf.placeholder(dtype=tf.float32)

        self.w1 = tf.Variable(tf.random_normal([layer_pr[0], layer_pr[1]]),dtype=tf.float32)
        self.b1 = tf.Variable(tf.random_normal([layer_pr[1]]),dtype=tf.float32)
        self.out1 = tf.add(tf.matmul(self.x,self.w1), self.b1)
        self.out1_ = tf.nn.relu(self.out1)
        self.out1_do = tf.nn.dropout(self.out1_,keep_prob=self.keep_prob)

        self.w2 = tf.Variable(tf.random_normal([layer_pr[1], layer_pr[2]]),dtype=tf.float32)
        self.b2 = tf.Variable(tf.random_normal([layer_pr[2]]),dtype=tf.float32)
        self.out2 = tf.add(tf.matmul(self.out1_,self.w2), self.b2)
        self.out2_ = tf.nn.relu(self.out2)
        self.out2_do = tf.nn.dropout(self.out2_,keep_prob=self.keep_prob)

        self.w3 = tf.Variable(tf.random_normal([layer_pr[2], layer_pr[3]]), dtype=tf.float32)
        self.b3 = tf.Variable(tf.random_normal([layer_pr[3]]), dtype=tf.float32)
        self.out3 = tf.add(tf.matmul(self.out2_,self.w3), self.b3)
        self.y = self.out3





        # for i in range(len(layer_pr)-1):
        #     self.weight_.append(tf.Variable(tf.zeros([layer_pr[i], layer_pr[i+1]]),dtype=tf.float32))
        #     self.bias_.append(tf.Variable(tf.zeros([layer_pr[i+1]]),dtype=tf.float32))

        # self.outlayer = []
        # self.outlayer.append(self.x)
        # for i in range(len(layer_pr)-2):
        #     temp_2 = tf.nn.relu(tf.add(tf.matmul(self.outlayer[-1], self.weight_[i]), self.bias_[i]))
        #     temp_ = tf.nn.dropout (temp_2, self.keep_prob_)
        #     self.outlayer.append(temp_)

        # predict
        # self.outlayer.append(tf.add(tf.matmul(self.outlayer[-1], self.weight_[len(layer_pr)-2]), self.bias_[len(layer_pr)-2]))
        # self.y = self.outlayer[-1]
        # self.cross_entropy = -tf.reduce_sum(self.y_ * tf.log(self.y))

        # cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.y, labels=self.y_))
        # self.train_step = tf.train.GradientDescentOptimizer
        self.train_step = tf.train.AdamOptimizer(self.meta_json["LEARNING_RATE"]).minimize(self.loss)

        # self.train_step = tf.train.GradientDescentOptimizer
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        # prediction
        self.ccc = 0
        self.saver = tf.train.Saver()

        return

    def train(self, x_train, y_train):
        channel_index = (4,5,9,17,57)
        x_train = x_train[:,channel_index,:,:]
        if self.meta_json["rest"] is True:
            self.initialize_variables()
            self.restoring_variables()
        else:
            self.initialize_variables()



        x_tt = x_train.reshape((x_train.shape[0], -1) ).astype(np.float32)
        # x_tt = tf.Variable( tf.cast( tf.reshape(x_train, (x_train.shape[0], -1) ), dtype=tf.float32) )
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

        for i in range(int(self.meta_json["NUM_STEPS"])):
            self.sess.run(self.train_step, feed_dict = {self.x: x_tt, self.y_: y, self.keep_prob:self.keep_prob_ })
            if i%(1000) == 0:
                self.ccc += 1
                print('[{0}/{1}] : {2} -- loss : {3}'.format(self.ccc , int(self.meta_json["NUM_STEPS"]/1000) , self.accuracy_score( x_train, y_train ), self.sess.run(self.loss, feed_dict = {self.x: x_tt, self.y_: y , self.keep_prob:self.keep_prob_ }) ) )
                try:
                    save_path = self.saver.save(self.sess, self.meta_json["MODEL_NAME"] + '-' + self.meta_json[
                        "STARTED_DATESTRING"] + "/model.ckpt")
                    save_path = self.saver.save(self.sess, self.meta_json["MODEL_NAME"] + '-' + self.meta_json[
                        "STARTED_DATESTRING"] + "/model" + str(i) + "iter.ckpt")
                    summary = self.sess.run(merged, feed_dict={self.x: x_tt, self.y_: y, self.keep_prob: 1.0 })
                    self.writer.add_summary(summary, i)
                except:
                    print("checkpoint didnt saved. pleas be sure that is well saved")

                self.writer.close()
        return 0


    def accuracy_score(self, x_test, y_test):
        if x_test.shape[1] >10:
            channel_index = (4, 5, 9, 17, 57)
            x_test=x_test[:,channel_index,:,:]
        # x_tt = tf.Variable(tf.reshape(x_test, (x_test.shape[0], -1)), dtype=tf.float32)
        x_tt = x_test.reshape((x_test.shape[0], -1)).astype(np.float32)
        # self.out_softmax = tf.nn.softmax()
        y_mod = np.zeros((y_test.shape[0],2), dtype=np.int16)
        y_test = y_test.astype(np.float32)
        for i in range(y_test.shape[0]):
            y_mod[i, int(((y_test[i]) + 1)/2)] = 1
        # y = tf.Variable(y_mod,dtype=tf.float32)
        y__ = y_mod.astype(np.float32)
        y_pred = np.argmax(self.sess.run(self.y, feed_dict={self.x: x_tt, self.keep_prob: 1.0 }),axis=1)
        self.acc = np.average(np.equal(y_pred, np.argmax(y__, axis = 1) ))

        return self.acc

class CNN1D :
    # origianl input data size : 61x61x64(frequency x time point x channel number) to 1D image
    # reference : A novel deep learning approach for classification of EEG motor imagerty signals, Yousef Razael Tabar and Ugur Halici, 2017, Journal of Neural Engineering
    #
    def __init__(self, meta_json):
        input_layer = 3
        self.meta_json = meta_json
        self.ccc = 0

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



class CNN2D(CNN1D) :
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
        y_pred = np.argmax(self.sess.run(self.y, feed_dict={self.x_in: x_tt}),axis=1)
        self.acc = np.average(np.equal(y_pred, np.argmax(y__, axis = 1) ))

        return self.acc

class CNN2D_shrinked(CNN2D):
    # need to be fixed [2018.02.05]
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


        channel_index = (4, 9, 17, 57)
        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        # conv network
        if self.meta_json["GPU"] is True:
            devicemapping = '/gpu:0'
        else :
            devicemapping = '/cpu:0'

        # conv network
        self.x_in = tf.placeholder(shape=[None, 61 * 2, 61 * 2, 1], dtype=tf.float32)
        self.y_ = tf.placeholder(dtype=tf.float32, shape=[None, 2])
        with tf.device(devicemapping):

            w1 = tf.Variable(tf.truncated_normal(shape=(7, 7, 1, 10), stddev=0.1))
            bias1 = tf.Variable(tf.constant(0.1, shape=[10]))
            conv1 = tf.nn.conv2d(self.x_in, w1, strides=[1, 2, 2, 1], padding="VALID")  # (?, 116, 116, 16)
            conv1_out = tf.nn.relu(conv1 + bias1)
            pool1 = tf.nn.max_pool(conv1_out, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')  # (?, 29, 29, 10)

            w2 = tf.Variable(tf.truncated_normal(shape=(7, 7, 10, 30), stddev=0.1))
            bias2 = tf.Variable(tf.constant(0.1, shape=[30]))
            conv2 = tf.nn.conv2d(pool1, w2, strides=[1, 1, 1, 1], padding="VALID")
            conv2_out = tf.nn.relu(conv2 + bias2)
            pool2 = tf.nn.max_pool(conv2_out, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')  # (?, 12, 12, 32)




        # This model cannot be structurally deep since this model applied 1d convolution
        self.out_layer = tf.reshape(pool2, shape=(-1, 12*12*30))
        # This model cannot be structurally deep since this model applied 1d convolution
        self.weight_ = tf.Variable(tf.zeros((12*12*30, 2), dtype=tf.float32), dtype=tf.float32)
        self.bias_ = tf.Variable(tf.zeros((2), dtype=tf.float32), dtype=tf.float32)

        self.y = tf.add(tf.matmul(self.out_layer, self.weight_), self.bias_)


        # cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.y, labels=self.y_))

        self.train_step = tf.train.AdamOptimizer(learning_rate=self.meta_json["LEARNING_RATE"]).minimize(self.loss)
        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()

        return


class CNN2DwithCAM(CNN2D) :
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


        channel_index = (4, 9, 17, 57)
        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        # conv network
        # if self.meta_json["GPU"] is True:
        #     devicemapping = '/gpu:0'
        # else :
        #     devicemapping = '/cpu:0'

        # conv network
        self.x_in = tf.placeholder(shape=[None, 61 * 2, 61 * 2, 1], dtype=tf.float32)
        self.y_ = tf.placeholder(dtype=tf.float32, shape=[None, 2])
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


        # cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
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
        x_tt = np.concatenate((temp1,temp2),axis=2).reshape((x_test.shape[0],61*2,61*2,1))

        y_mod = np.zeros((y_test.shape[0],2), dtype=np.int16)
        y_test = y_test.astype(np.float32)
        for i in range(y_test.shape[0]):
            y_mod[i, int(((y_test[i]) + 1)/2)] = 1
        # y = tf.Variable(y_mod,dtype=tf.float32)
        y = y_mod.astype(np.float32)

        weights_ = self.sess.run(self.weight_)
        pool2seed = self.sess.run(self.pool2, feed_dict={self.x_in: x_tt})  # batch,12,12,30
        self.image = np.zeros((12, 12, 2))
        for bi in range(pool2seed.shape[0]):
            for i in range(self.weight_.shape[0]):
                self.image[:, :, np.where(y[bi] == 1)[0].tolist()[0]] += (pool2seed[bi,:,:,:].reshape((12,12,30))[:,:,i] * weights_ [i,np.where(y[bi] == 1)[0].tolist()[0]])

        # devide im 1000
        self.image = self.image - np.min(self.image)
        denom = np.max(self.image)/1000
        self.image = np.int16(self.image/denom)

        np.save(self.meta_json["MODEL_NAME"] + '-' + self.meta_json[
                        "STARTED_DATESTRING"] + '/CAM_{0}'.format(self.ccc) ,self.image)

        return 0


    def train(self, x_train, y_train, x_test, y_test):

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
            totbat = x_tt.shape[0]
            fortemp = 1
            while (totbat/fortemp) > 20:
                fortemp += 1
            e_bat =int(totbat / fortemp)
            for j in range(fortemp):
                self.sess.run(self.train_step, feed_dict={self.x_in: x_tt[e_bat * j:e_bat * (j + 1), :, :, :],
                                                          self.y_: y[e_bat * j:e_bat * (j + 1), :]})
            if i%(1000) == 0:
                self.ccc += 1
                print('[{0}/{1}] : {2} -- loss : {3}'.format(self.ccc , int(self.meta_json["NUM_STEPS"]/1000) , self.accuracy_score(x_test, y_test), self.sess.run(self.loss, feed_dict = {self.x_in: x_tt, self.y_: y}) ) )
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


class CNN3D_TFC :
    # not the freq as 61, 30
    def __init__(self, meta_json):
        input_layer = 3
        self.meta_json = meta_json
        self.ccc = 0

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
            fortemp = 1
            while (totbat/fortemp) > 20:
                fortemp += 1
            e_bat =int(totbat / fortemp)
            for j in range(fortemp):
                self.sess.run(self.train_step, feed_dict = {self.x_in: x_train[e_bat*j:e_bat*(j+1),:,:,:,:], self.y_: y[e_bat*j:e_bat*(j+1),:]})
            if i%(1000) == 0:
                self.ccc += 1
                totbat_ = x_test.shape[0]
                fortemp_ = 1
                while (totbat_ / fortemp_) > 20:
                    fortemp_ += 1
                e_bat_ = int(totbat_ / fortemp_)
                dap = []
                for k in range(fortemp_):
                    dap.append(self.accuracy_score(x_test[e_bat*k:e_bat*(k+1),:,:,:], y_test[e_bat*k:e_bat*(k+1)]))
                ls =  0
                for j in range(fortemp):
                    ls += self.sess.run(self.loss, feed_dict={self.x_in: x_train[e_bat*j:e_bat*(j+1),:,:,:,:], self.y_: y[e_bat*j:e_bat*(j+1),:]})


                print('[{0}/{1}] : {2} -- loss : {3}'.format(self.ccc , int(self.meta_json["NUM_STEPS"]/1000) , np.mean(dap), ls) )
                try:
                    # self.CAMgen(x_test,y_test)
                    save_path = self.saver.save(self.sess, self.meta_json["MODEL_NAME"] + '-' + self.meta_json[
                        "STARTED_DATESTRING"] + "/model.ckpt")
                    save_path = self.saver.save(self.sess, self.meta_json["MODEL_NAME"] + '-' + self.meta_json[
                        "STARTED_DATESTRING"] + "/model" + str(i) + "iter.ckpt")
                    summary = self.sess.run(merged, feed_dict={self.x_in: x_train[e_bat*j:e_bat*(j+1),:,:,:,:], self.y_: y[e_bat*j:e_bat*(j+1),:]})
                    self.writer.add_summary(summary, i)
                except:
                    print("checkpoint didnt saved. pleas be sure that is well saved")

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
        self.x_in = tf.placeholder(shape = [None, 30, 30, 30, 1], dtype=tf.float32)
        self.y_ = tf.placeholder(dtype=tf.float32, shape = [None, 2])

        # with tf.device('/cpu:0'):
        w1 = tf.Variable(tf.truncated_normal(shape = (7,7,7,1,10) , stddev=0.1))
        conv1 = tf.nn.conv3d(self.x_in, w1, strides = [1,1,1,1,1], padding="VALID") # (?, 116, 116, 16)
        conv1_out = tf.nn.relu(conv1) # (?, 116, 116, 16)
        pool1 = tf.nn.max_pool3d(conv1_out, ksize= [1,3,3,3,1], strides=[1,1,1,1,1], padding='SAME')# (?, 7,6,6,20)

        w11 = tf.Variable(tf.truncated_normal(shape = (7,7,7,10,30) , stddev=0.1))
        conv11 = tf.nn.conv3d(pool1, w11, strides = [1,1,1,1,1], padding="VALID") # (?, 116, 116, 16)
        conv11_out = tf.nn.relu(conv11) # (?, 116, 116, 16)
        self.pool2 = tf.nn.max_pool3d(conv11_out, ksize= [1,3,3,3,1], strides=[1,2,2,2,1], padding='SAME')# (?, 9,9,9,30)


        # This model cannot be structurally deep since this model applied 1d convolution
        self.gap = tf.reduce_mean(tf.reshape(self.pool2, shape=(-1,9**3,30)), axis=1)

        self.weight_ = tf.Variable(tf.zeros((30, 2), dtype=tf.float32), dtype=tf.float32)

        self.out_layer = self.gap

        self.y = tf.matmul(self.out_layer, self.weight_)

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
            fortemp = 1
            while (totbat/fortemp) > 20:
                fortemp += 1
            e_bat =int(totbat / fortemp)
            for j in range(fortemp):
                self.sess.run(self.train_step, feed_dict = {self.x_in: x_train[e_bat*j:e_bat*(j+1),:,:,:,:], self.y_: y[e_bat*j:e_bat*(j+1),:]})
            if i%(1000) == 0:
                self.ccc += 1
                totbat_ = x_test.shape[0]
                fortemp_ = 1
                while (totbat_ / fortemp_) > 20:
                    fortemp_ += 1
                e_bat_ = int(totbat_ / fortemp_)
                dap = []
                for k in range(fortemp_):
                    dap.append(self.accuracy_score(x_test[e_bat*k:e_bat*(k+1),:,:,:], y_test[e_bat*k:e_bat*(k+1)]))
                ls =  0
                for j in range(fortemp):
                    ls += self.sess.run(self.loss, feed_dict={self.x_in: x_train[e_bat*j:e_bat*(j+1),:,:,:,:], self.y_: y[e_bat*j:e_bat*(j+1),:]})


                print('[{0}/{1}] : {2} -- loss : {3}'.format(self.ccc , int(self.meta_json["NUM_STEPS"]/1000) , np.mean(dap), ls) )
                try:
                    self.CAMgen(x_test,y_test)
                    save_path = self.saver.save(self.sess, self.meta_json["MODEL_NAME"] + '-' + self.meta_json[
                        "STARTED_DATESTRING"] + "/model.ckpt")
                    save_path = self.saver.save(self.sess, self.meta_json["MODEL_NAME"] + '-' + self.meta_json[
                        "STARTED_DATESTRING"] + "/model" + str(i) + "iter.ckpt")
                    summary = self.sess.run(merged, feed_dict={self.x_in: x_train[e_bat*j:e_bat*(j+1),:,:,:,:], self.y_: y[e_bat*j:e_bat*(j+1),:]})
                    self.writer.add_summary(summary, i)
                except:
                    print("checkpoint didnt saved. pleas be sure that is well saved")

        self.writer.close()

        return 0


    def CAMgen(self, x_test, y_test):
        # none 12 12 30
        # num_feature = self.weight_.shape[0]
        # num_classes = self.weight_.shape[1]
        ch = np.array([63,62,64,17,10,5,1,61,8,11,2,18,13,9,6,3,59,58,12,60,19,14,57,56,23,15,7,54,53,55])
        channel_index = tuple((ch-1).tolist())
        x_test = x_test.reshape((x_test.shape[0],x_test.shape[1],x_test.shape[2],x_test.shape[3],1))
        x_trains = x_test[:, channel_index, :, :30, :]
        x_test = np.zeros((x_test.shape[0], 30, 30, x_trains.shape[3], 1)) # ? 64 30 30 1
        for tr in range(x_test.shape[2]):
            x_test[:,:,tr,:,:] = (x_trains[:,:,2*tr,:,:] +x_trains[:,:,2*tr+1,:,:]) /2

        y_mod = np.zeros((y_test.shape[0],2), dtype=np.int16)
        y_test = y_test.astype(np.float32)
        for i in range(y_test.shape[0]):
            y_mod[i, int(((y_test[i]) + 1)/2)] = 1
        # y = tf.Variable(y_mod,dtype=tf.float32)
        y = y_mod.astype(np.float32)

        weights_ = self.sess.run(self.weight_) # 30,2
        pool2seed = self.sess.run(self.pool2, feed_dict={self.x_in: x_test})  # batch,9,9,9,30
        self.image = np.zeros((9, 9, 9, 2))
        for bi in range(pool2seed.shape[0]):
            for i in range(self.weight_.shape[0]):
                self.image[:, :, :, np.where(y[bi] == 1)[0].tolist()[0]] += (pool2seed[bi,:,:,:,:].reshape((9,9,9,30))[:,:,:,i] * weights_ [i,np.where(y[bi] == 1)[0].tolist()[0]])

        # devide im 1000
        self.image = self.image - np.min(self.image)
        denom = np.max(self.image)/1000
        self.image = np.int16(self.image/denom)

        np.save(self.meta_json["MODEL_NAME"] + '-' + self.meta_json[
                        "STARTED_DATESTRING"] + '/CAM_{0}'.format(self.ccc) ,self.image)

        return 0


if __name__ == '__dmain__':
    meta_json=dict()
    meta_json["NUM_STEPS"] = 1e4
    meta_json["LEARNING_RATE"] = 5e-2
    model1 = CNN1D(meta_json)

    x_train = np.random.rand(1000, 64, 61, 61)
    x_train[:500,:,:,:]= x_train[:500,:,:,:]+2
    y_train = np.ones((1000, ))
    y_train[:500]*=-1

    x_tt = np.zeros((x_train.shape[0], 61 * 64, 61, 1))
    for i in range(x_train.shape[3]):
        x_tt[:, :, i, :] = x_train[:, :, :, i].reshape((-1, x_train.shape[2] * x_train.shape[1], 1))

    # conv network
    x_in = tf.placeholder(shape=[None, 3904, 61, 1], dtype=tf.float32)
    w1 = tf.Variable(tf.truncated_normal(shape=(61 * 64, 7, 1, 16), stddev=0.1))
    bias1 = tf.Variable(tf.constant(0.1, shape=[16]))
    conv1 = tf.nn.conv2d(x_in, w1, strides=[1, 1, 1, 1], padding="SAME")
    conv1_out = tf.nn.relu(conv1 + bias1)
    pool1 = tf.nn.max_pool(conv1_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    w2 = tf.Variable(tf.truncated_normal(shape=(5, 5, 16, 32), stddev=0.1))
    bias2 = tf.Variable(tf.constant(0.1, shape=[32]))
    conv2 = tf.nn.conv2d(pool1, w2, strides=[1, 1, 1, 1], padding="SAME")
    conv2_out = tf.nn.relu(conv2 + bias2)
    pool2 = tf.nn.max_pool(conv2_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(conv1, feed_dict={x_in: x_tt.astype(np.float32)})

if __name__ == '__main__':
    meta_json=dict()
    meta_json["NUM_STEPS"] = 1e4
    meta_json["LEARNING_RATE"] = 5e-2
    meta_json["rest"] = False
    meta_json["MODEL_NAME"]="CNN3D_TFC_CAM"
    meta_json["STARTED_DATESTRING"]="2018-01-29"
    meta_json["LOGDIR_ROOT"]="LOG"
    meta_json["GPU"] = False
    model1 = CNN3D_TFC_CAM(meta_json)
    x_train = np.concatenate( (np.ones((10,64,61,61)),  np.ones((10,64,61,61)) *(-2)), axis=0)
    y_train = np.concatenate( (np.ones(10), np.ones(10)*(-1)) ,axis = 0)
    y_mod = np.zeros((y_train.shape[0], 2), dtype=np.int16)
    y_train = y_train.astype(np.float32)
    for i in range(y_train.shape[0]):
        y_mod[i, int(((y_train[i]) + 1) / 2)] = 1



    model1.train(x_train, y_train, x_train, y_train)
