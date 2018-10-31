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

    def restoring_variables(self):
        som = [f for f in os.listdir(self.meta_json["MODEL_NAME"] + '-'  + self.meta_json["STARTED_DATESTRING"] + '/.') if f.lower().endswith('iter.ckpt')]
        soi = []
        for i in som:
            soi.append(int(i.split('.')[0][5:-4]))
        self.saver.restore(self.sess, self.meta_json["MODEL_NAME"] + '-'  + self.meta_json["STARTED_DATESTRING"] + "/model.ckpt")
        self.ccc = 100

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

        self.sess = tf.Session()
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

        self.writer = tf.summary.FileWriter(
            './' + self.meta_json["MODEL_NAME"] + self.meta_json["STARTED_DATESTRING"] + self.meta_json["LOGDIR_ROOT"],
            self.sess.graph)

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
        self.sess = tf.Session()

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
        elif self.meta_json['addto'] == 'CNN2DwithCAM':
            self.model = eval(self.meta_json['addto'] + '(self.meta_json)')
            self.model.initialize_variables()
            self.model.restoring_variables()
        else:
            print('BOOM in half')
        return

    def puts_out(self, spectro_train, spectro_test):
        if self.meta_json['addto'] == 'CNN2DwithCAM':
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

        # RNN 은 순서가 있는 자료를 다루므로,
        # 한 번에 입력받는 갯수와, 총 몇 단계로 이루어져있는 데이터를 받을지를 설정해야합니다.
        # 이를 위해 가로 픽셀수를 n_input 으로, 세로 픽셀수를 입력 단계인 n_step 으로 설정하였습니다.
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


if __name__ == '__main__':
    meta_json=dict()
    meta_json["NUM_STEPS"] = 1e4
    meta_json["LEARNING_RATE"] = 5e-2
    meta_json["rest"] = False
    meta_json["MODEL_NAME"]="CNN2DwithCAM"
    meta_json["STARTED_DATESTRING"]="1st"
    meta_json["LOGDIR_ROOT"]="LOG"
    meta_json["GPU"] = False
    model_LSTM = addon_LSTM(meta_json)
    x_train = np.concatenate( (np.ones((10,64,61,61)),  np.ones((10,64,61,61)) *(-2)), axis=0)
    y_train = np.concatenate( (np.ones(10), np.ones(10)*(-1)) ,axis = 0)
    y_mod = np.zeros((y_train.shape[0], 2), dtype=np.int16)
    y_train = y_train.astype(np.float32)
    for i in range(y_train.shape[0]):
        y_mod[i, int(((y_train[i]) + 1) / 2)] = 1

    model_LSTM.train(x_train, y_train, x_train, y_train)
