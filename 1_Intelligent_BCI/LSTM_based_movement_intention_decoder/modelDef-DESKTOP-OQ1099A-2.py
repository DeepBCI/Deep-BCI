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
        self.keep_prob_ = tf.placeholder(dtype=tf.float32)

        self.w1 = tf.Variable(tf.random_normal([layer_pr[0], layer_pr[1]]),dtype=tf.float32)
        self.b1 = tf.Variable(tf.random_normal([layer_pr[1]]),dtype=tf.float32)
        self.out1 = tf.add(tf.matmul(self.x,self.w1), self.b1)
        self.out1_ = tf.nn.relu(self.out1)
        self.out1_do = tf.nn.dropout(self.out1_,keep_prob=self.keep_prob_)

        self.w2 = tf.Variable(tf.random_normal([layer_pr[1], layer_pr[2]]),dtype=tf.float32)
        self.b2 = tf.Variable(tf.random_normal([layer_pr[2]]),dtype=tf.float32)
        self.out2 = tf.add(tf.matmul(self.out1_,self.w2), self.b2)
        self.out2_ = tf.nn.relu(self.out2)
        self.out2_do = tf.nn.dropout(self.out2_,keep_prob=self.keep_prob_)

        self.w3 = tf.Variable(tf.random_normal([layer_pr[2], layer_pr[4]]), dtype=tf.float32)
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
            self.sess.run(self.train_step, feed_dict = {self.x: x_tt, self.y_: y, self.keep_prob_:self.keep_prob })
            if i%(1000) == 0:
                self.ccc += 1
                print('[{0}/{1}] : {2} -- loss : {3}'.format(self.ccc , int(self.meta_json["NUM_STEPS"]/1000) , self.accuracy_score( x_train, y_train ), self.sess.run(self.loss, feed_dict = {self.x: x_tt, self.y_: y , self.keep_prob_:self.keep_prob }) ) )
                try:
                    save_path = self.saver.save(self.sess, self.meta_json["MODEL_NAME"] + '-' + self.meta_json[
                        "STARTED_DATESTRING"] + "/model.ckpt")
                    save_path = self.saver.save(self.sess, self.meta_json["MODEL_NAME"] + '-' + self.meta_json[
                        "STARTED_DATESTRING"] + "/model" + str(i) + "iter.ckpt")
                    summary = self.sess.run(merged, feed_dict={self.x: x_tt, self.y_: y, self.keep_prob_: 1.0 })
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
        y_pred = np.argmax(self.sess.run(self.y, feed_dict={self.x: x_tt, self.keep_prob_: 1.0 }),axis=1)
        self.acc = np.average(np.equal(y_pred, np.argmax(y__, axis = 1) ))

        return self.acc

class CNN1D :
    # origianl input data size : 61x61x64(frequency x time point x channel number) to 1D image
    # reference : A novel deep learning approach for classification of EEG motor imagerty signals, Yousef Razael Tabar and Ugur Halici, 2017, Journal of Neural Engineering
    #
    def __init__(self, meta_json):
        input_layer = 3
        self.meta_json = meta_json

        return

    def initialize_variables(self):
        # reshape x_train to 2D image
        # For this case, use DNN with 5 channels only : E6(Fz), E10(Fp1), E5(Fp2), E18(F7), E58(F8)
        # x_train = np.zeros((100,64,61,61))
        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

        channel_index = (4, 5, 9, 17, 57)

        # conv network
        x_in = tf.placeholder(shape = [None, 61*5, 61, 1], dtype=tf.float32)
        self.y_ = tf.placeholder(dtype=tf.float32, shape = [None, 2])

        if self.meta_json["GPU"] is True:
            devicemapping = '/gpu:0'
        else :
            devicemapping = '/cpu:0'


        with tf.device(devicemapping):
            w1 = tf.Variable(tf.truncated_normal(shape = (61*5,7,1,60) , stddev=0.1))
            bias1 = tf.Variable(tf.constant(0.1, shape = [60]))
            conv1 = tf.nn.conv2d(x_in, w1, strides = [1,1,1,1], padding="VALID")
            conv1_out = tf.nn.relu(conv1 + bias1)
            pool1 = tf.nn.max_pool(conv1_out, ksize= [1,1,5,1], strides=[1,1,5,1], padding='SAME')

            # This model cannot be structurally deep since this model applied 1d convolution
            self.weight_ = tf.Variable(tf.zeros((660,2),dtype=tf.float32),dtype=tf.float32)
            self.bias_ = tf.Variable(tf.zeros((2),dtype=tf.float32),dtype=tf.float32)

        self.y = tf.add(tf.matmul(pool1, self.weight_) + self.bias_)

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
        self.saver.restore(self.sess, self.meta_json["MODEL_NAME"] + '-'  + self.meta_json["STARTED_DATESTRING"] + "/model" +str(max(soi)) + ".ckpt")
        self.ccc = max(soi)

        return


    def train(self, x_train, y_train):

        if self.meta_json["rest"] is True:
            self.initialize_variables()
            self.restoring_variables()
        else:
            self.initialize_variables()

        channel_index = (4, 5, 9, 17, 57)
        x_train = x_train[:,channel_index,:,:]
        x_tt = np.zeros((x_train.shape[0], 61*5, 61, 1))
        for i in range(x_train.shape[3]):
            x_tt [:,:,i,:] = x_train[:,:,:,i].reshape((-1,x_train.shape[2]*x_train.shape[1], 1))

        y_mod = np.zeros((y_train.shape[0],2), dtype=np.int16)
        y_train = y_train.astype(np.float32)
        for i in range(y_train.shape[0]):
            y_mod[i, int(((y_train[i]) + 1)/2)] = 1
        # y = tf.Variable(y_mod,dtype=tf.float32)
        y = y_mod.astype(np.float32)


        loss_ = sess.run(self.sess.run(self.loss, feed_dict={self.x: x_tt, self.y_: y}))
        loss_hist = tf.summary.scalar('loss', loss_)
        merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(
            './' + self.meta_json["MODEL_NAME"] + self.meta_json["STARTED_DATESTRING"] + self.meta_json["LOGDIR_ROOT"],
            self.sess.graph)

        x_tt = x_train.reshape((x_train.shape[0], -1) ).astype(np.float32)
        # x_tt = tf.Variable( tf.cast( tf.reshape(x_train, (x_train.shape[0], -1) ), dtype=tf.float32) )
        y_mod = np.zeros((y_train.shape[0],2), dtype=np.int16)
        y_train = y_train.astype(np.float32)
        for i in range(y_train.shape[0]):
            y_mod[i, int(((y_train[i]) + 1)/2)] = 1
        # y = tf.Variable(y_mod,dtype=tf.float32)
        y = y_mod.astype(np.float32)
        for i in range(int(self.meta_json["NUM_STEPS"])):
            self.sess.run(self.train_step, feed_dict = {self.x: x_tt, self.y_: y})
            if i%(1000) == 0:
                self.ccc += 1
                print('[{0}/{1}] : {2} -- loss : {3}'.format(self.ccc , int(self.meta_json["NUM_STEPS"]/1000) , self.accuracy_score( x_tt, y), self.sess.run(self.loss, feed_dict = {self.x: x_tt, self.y_: y}) ) )
                try:
                    save_path = self.saver.save(self.sess, self.meta_json["MODEL_NAME"] + '-' + self.meta_json[
                        "STARTED_DATESTRING"] + "/model.ckpt")
                    save_path = self.saver.save(self.sess, self.meta_json["MODEL_NAME"] + '-' + self.meta_json[
                        "STARTED_DATESTRING"] + "/model" + str(i) + "iter.ckpt")
                    self.writer.add_summary(loss_, i)
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
        y_pred = np.argmax(self.sess.run(self.y, feed_dict={self.x: x_tt}),axis=1)
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

        channel_index = (4, 5, 9, 17, 57)

        # conv network
        x_in = tf.placeholder(shape = [None, 61*5, 61, 1], dtype=tf.float32)
        self.y_ = tf.placeholder(dtype=tf.float32, shape = [None, 2])

        if self.meta_json["GPU"] is True:
            devicemapping = '/gpu:0'
        else :
            devicemapping = '/cpu:0'



        # conv network
        self.x_in = tf.placeholder(shape = [None, 61*2, 61*2, 1], dtype=tf.float32)
        self.y_ = tf.placeholder(dtype=tf.float32, shape = [None, 2])

        with tf.device(devicemapping):

            w1 = tf.Variable(tf.truncated_normal(shape = (7,7,1,16) , stddev=0.1))
            bias1 = tf.Variable(tf.constant(0.1, shape = [16]))
            conv1 = tf.nn.conv2d(x_in, w1, strides = [1,1,1,1], padding="VALID") # (?, 116, 116, 16)
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


        # This model cannot be structurally deep since this model applied 1d convolution
        self.weight_ = tf.Variable(tf.zeros((1024,2),dtype=tf.float32),dtype=tf.float32)
        self.bias_ = tf.Variable(tf.zeros((2),dtype=tf.float32),dtype=tf.float32)


        self.y = tf.add(tf.matmul(pool4, self.weight_) + self.bias_)

        # cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.y, labels=self.y_))

        self.train_step = tf.train.AdamOptimizer(learning_rate=self.meta_json["LEARNING_RATE"]).minimize(self.loss)

        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()

        return

    def restoring_variables(self):
        self.saver = tf.train.Saver()
        self.sess =tf.Session()
        som = [f for f in os.listdir(self.meta_json["MODEL_NAME"] + '-'  + self.meta_json["STARTED_DATESTRING"] + '/.') if f.lower().endswith('iter.ckpt')]
        soi = []
        for i in som:
            soi.append(int(i.split('.')[0][5:-4]))
        self.saver.restore(self.sess, self.meta_json["MODEL_NAME"] + '-'  + self.meta_json["STARTED_DATESTRING"] + "/model" +str(max(soi)) + ".ckpt")
        self.ccc = max(soi)

        return


    def train(self, x_train, y_train):

        if self.meta_json["rest"] is True:
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


        loss_ = sess.run(self.sess.run(self.loss, feed_dict={self.x_in: x_tt, self.y_: y}))
        loss_hist = tf.summary.scalar('loss', loss_)
        merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(
            './' + self.meta_json["MODEL_NAME"] + self.meta_json["STARTED_DATESTRING"] + self.meta_json["LOGDIR_ROOT"],
            self.sess.graph)

        x_tt = x_train.reshape((x_train.shape[0], -1) ).astype(np.float32)
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
                print('[{0}/{1}] : {2} -- loss : {3}'.format(self.ccc , int(self.meta_json["NUM_STEPS"]/1000) , self.accuracy_score(x_tt, y), self.sess.run(self.loss, feed_dict = {self.x_in: x_tt, self.y_: y}) ) )
                try:
                    save_path = self.saver.save(self.sess, self.meta_json["MODEL_NAME"] + '-' + self.meta_json[
                        "STARTED_DATESTRING"] + "/model.ckpt")
                    save_path = self.saver.save(self.sess, self.meta_json["MODEL_NAME"] + '-' + self.meta_json[
                        "STARTED_DATESTRING"] + "/model" + str(i) + "iter.ckpt")
                    self.writer.add_summary(loss_, i)
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
        y__ = y_mod.astype(np.float32)
        y_pred = np.argmax(self.sess.run(self.y, feed_dict={self.x_in: x_tt}),axis=1)
        self.acc = np.average(np.equal(y_pred, np.argmax(y__, axis = 1) ))

        return self.acc


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
        if self.meta_json["GPU"] is True:
            devicemapping = '/gpu:0'
        else :
            devicemapping = '/cpu:0'

        # conv network
        self.x_in = tf.placeholder(shape=[None, 61 * 2, 61 * 2, 1], dtype=tf.float32)
        self.y_ = tf.placeholder(dtype=tf.float32, shape=[None, 2])
        with tf.device(devicemapping):

            w1 = tf.Variable(tf.truncated_normal(shape=(7, 7, 1, 16), stddev=0.1))
            bias1 = tf.Variable(tf.constant(0.1, shape=[16]))
            conv1 = tf.nn.conv2d(self.x_in, w1, strides=[1, 1, 1, 1], padding="VALID")  # (?, 116, 116, 16)
            conv1_out = tf.nn.relu(conv1 + bias1)  # (?, 116, 116, 16)
            pool1 = tf.nn.max_pool(conv1_out, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')  # (?, 58, 58, 16)

            w2 = tf.Variable(tf.truncated_normal(shape=(7, 7, 16, 32), stddev=0.1))
            bias2 = tf.Variable(tf.constant(0.1, shape=[32]))
            conv2 = tf.nn.conv2d(pool1, w2, strides=[1, 1, 1, 1], padding="VALID")  # (?, 52, 52, 32)
            conv2_out = tf.nn.relu(conv2 + bias2)  # (?, 52, 52, 32)
            pool2 = tf.nn.max_pool(conv2_out, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')  # (?, 26, 26, 32)

            w3 = tf.Variable(tf.truncated_normal(shape=(5, 5, 32, 48), stddev=0.1))
            bias3 = tf.Variable(tf.constant(0.1, shape=[48]))
            conv3 = tf.nn.conv2d(pool2, w3, strides=[1, 1, 1, 1], padding="VALID")  # (?, 22, 22, 48)
            conv3_out = tf.nn.relu(conv3 + bias3)  # (?, 22, 22, 48)
            pool3 = tf.nn.max_pool(conv3_out, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')  # (?, 11, 11, 48)

            w4 = tf.Variable(tf.truncated_normal(shape=(5, 5, 48, 64), stddev=0.1))
            bias4 = tf.Variable(tf.constant(0.1, shape=[64]))
            conv4 = tf.nn.conv2d(pool3, w4, strides=[1, 1, 1, 1], padding="VALID")  # (?, 7, 7, 64)
            conv4_out = tf.nn.relu(conv4 + bias4)  # (?, 7, 7, 64)
            pool4 = tf.nn.max_pool(conv4_out, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')  # (?, 4, 4, 64)

        # This model cannot be structurally deep since this model applied 1d convolution
        self.weight_ = tf.Variable(tf.zeros((1024, 2), dtype=tf.float32), dtype=tf.float32)
        self.bias_ = tf.Variable(tf.zeros((2), dtype=tf.float32), dtype=tf.float32)

        self.y = tf.add(tf.matmul(pool4, self.weight_) + self.bias_)

        # cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.y, labels=self.y_))

        self.train_step = tf.train.AdamOptimizer(learning_rate=self.meta_json["LEARNING_RATE"]).minimize(self.loss)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()

        return


class CNN3D :
    def __init__(self, meta_json):
        input_layer = 3

        # conv net 1
        conv1 = tf.layers.conv2d( inputs = input_layer,
                                  filters = 32,
                                  kernel_size = [5,5],
                                  padding= "same",
                                  activation = tf.nn.relu)



        return

    def train(self, x_train, y_train):
        return self.model.fit(x_train, y_train)

    def predict(self, x_test):
        estimation = self.model.predict(x_test)
        return estimation

    def accuracy_score(self, x_test, y_test):
        estimation = self.model.predict(x_test)
        return acc_func(estimation, y_test)


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
    meta_json["rest"]=False
    meta_json["MODEL_NAME"]="CNN1D"
    meta_json["STARTED_DATESTRING"]="2018-01-26"
    meta_json["LOGDIR_ROOT"]="LOG"
    meta_json["GPU"]=True
    model1 = CNN1D(meta_json)
    x_train = np.concatenate( (np.ones((1000,64,61,61)),  np.ones((1000,64,61,61)) *(-2)), axis=0)
    y_train = np.concatenate( (np.ones(1000), np.ones(1000)*(-1)) ,axis = 0)
    y_mod = np.zeros((y_train.shape[0], 2), dtype=np.int16)
    y_train = y_train.astype(np.float32)
    for i in range(y_train.shape[0]):
        y_mod[i, int(((y_train[i]) + 1) / 2)] = 1



    model1.train(x_train, y_train)


    # x_test =  np.random.rand(100, 61, 61)
    # meta_json = meta_json
    #
    #
    # x = tf.placeholder(dtype=tf.float32, shape=[None, 61 * 61])
    # y_ = tf.placeholder(dtype=tf.float32, shape=[None, 2])
    #
    #
    # weight_pre =tf.Variable(tf.zeros([61*61, 25]), dtype=tf.float32)
    # bias_pre =tf.Variable(tf.zeros([25]), dtype=tf.float32)
    # outlayer = tf.matmul(x, weight_pre) + bias_pre
    #
    # weight_ = tf.Variable(tf.zeros([25, 2]), dtype=tf.float32)
    # bias_ = tf.Variable(tf.zeros([2]), dtype=tf.float32)

    # weight_ = []
    # bias_ = []
    # layer_pr = [61 * 61, 100, 40, 15, 2]
    # for i in range(4):
    #     weight_.append(tf.Variable(tf.zeros([layer_pr[i], layer_pr[i + 1]]), dtype=tf.float32))
    #     bias_.append(tf.Variable(tf.zeros([layer_pr[i + 1]]), dtype=tf.float32))

    # outlayer = []
    # outlayer.append(x)
    # for i in range(3):
    #     temp_ = tf.nn.relu(tf.matmul(outlayer[-1], weight_[i]) + bias_[i])
    #     outlayer.append(temp_)
    #
    # # predict
    # y = tf.nn.softmax(tf.matmul(outlayer[-1], weight_[3]) + bias_[3])

    # cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

    # x = tf.placeholder(dtype=tf.float32, shape=[None, 61 * 61])
    # y_ = tf.placeholder(dtype=tf.float32, shape=[None, 2])
    #
    # predict
    # y = tf.nn.softmax(tf.matmul(outlayer, weight_) + bias_)
    # y = tf.matmul(outlayer, weight_) + bias_

    #cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits( logits = y , labels  = y_))
    # self.train_step = tf.train.GradientDescentOptimizer
    train_step = tf.train.AdamOptimizer(learning_rate=meta_json["LEARNING_RATE"]).minimize(loss)
    #train_step = tf.train.GradientDescentOptimizer(learning_rate=meta_json["LEARNING_RATE"]).minimize(cross_entropy)
    #
    # # prediction
    #
    #
    #
    #
    # sess = tf.Session()
    # ccc=0
    # sess.run(tf.global_variables_initializer())
    # for i in range(int(meta_json["NUM_STEPS"])):
    #     sess.run(train_step, feed_dict={x: x_train.reshape(x_train.shape[0], -1), y_: y_mod.astype(np.float32)})
    #     if i%(int(meta_json["NUM_STEPS"])/100) == 0:
    #         ccc += 1
    #         print('[{0}/100] : {1}'.format(ccc, sess.run(loss, feed_dict={x: x_train.reshape(x_train.shape[0], -1), y_: y_mod.astype(np.float32)})))
    #
    # sess.run(y, feed_dict={x: x_train.reshape(x_train.shape[0], -1)})
    #
    #

