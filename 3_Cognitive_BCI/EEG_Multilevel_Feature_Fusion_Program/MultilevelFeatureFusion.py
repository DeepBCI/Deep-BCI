from scipy import io
import tensorflow as tf
import numpy as np
import random
import time
import h5py
import math

## Parameters (User defined parameter)
data_train_path = './rap_load37_psd_fold.mat' # training data file location
data_test_path =  './rap_load37_psd_fold.mat' # test data file location
                                              # ** file format **
                                              # * file extension type should be "h5"
                                              # * file should be contain 'data' and 'dataIdx'
                                              # * data: 4D array [the number of trials x spatial x-axis x spatial y-axis x freqeuncy] e.g) [5301 x 32 x 32 x 20]
                                              # * dataIdx: one-hot encoded 2D array [the number of trials x the number of classes]    e.g) [5301 x 4]
save_path = './result/'                       # save model location
model_name = 'Multilevel_FUSION'              # save model name
learning_rate = 0.001                         # learning rate (0.001 is recommended)
decaying_epoch = 200                          # epoch of deacying learning rate (200 is recommended)
batch_size = 32                               # batch size in training step (32 is recommended)
training_epoch = 400                          # total training epoch ( 400 is recommended)
display_epoch_train = 1                       # epoch of displaying train accuracy and loss
display_epoch_test = 1                        # epoch of saving model and displaying test accuracy and loss

#######################################################################################################################
#######################################################################################################################
########################################################################################################################
########################################################################################################################
## Data loading
with h5py.File(data_train_path, 'r') as f:
    data_train = np.array(f['data'])
    dataIdx_train = np.array(f['dataIdx'])
    data_train = data_train.transpose()
    dataIdx_train = dataIdx_train.transpose()

with h5py.File(data_test_path, 'r') as f:
    data_test = np.array(f['data'])
    dataIdx_test = np.array(f['dataIdx'])
    data_test = data_test.transpose()
    dataIdx_test = dataIdx_test.transpose()

start_time = time.time()
print('data train path: ', data_train_path)
print('data test path: ', data_test_path)
print('save path: ', save_path)
########################################################################################################################
########################################################################################################################
## Network Parameters
input_shape = list(data_train.shape[1:])
n_samples = data_train.shape[0]
n_classes = dataIdx_train.shape[1]
graph = tf.get_default_graph()
########################################################################################################################
## 3 dimentional convolutional layer module
def conv3d(x, weight, ksize, kstrides=[1,1,1], activation='none', pool=False, psize=[2,2,2], pstrides=[2,2,2], padding='VALID',batch_norm = False, is_training = False, l2_losss = 0, l2_loss=False,init_counter = False):
    if init_counter:
        conv3d.counter = 0
    conv3d.counter += 1
    varname_conv = 'conv3d_conv' + str(conv3d.counter)
    varname_bias = 'conv3d_bias' + str(conv3d.counter)
    weight[varname_conv] = tf.get_variable(shape = ksize, name = varname_conv,initializer=tf.contrib.layers.xavier_initializer(uniform=False))
    weight[varname_bias] = tf.Variable(tf.zeros(ksize[4]), name= varname_bias)

    x = tf.nn.conv3d(x, weight[varname_conv], strides=[1] + kstrides + [1], padding=padding)
    x = tf.nn.bias_add(x, weight[varname_bias])

    if batch_norm:
        x = tf.contrib.layers.batch_norm(x, is_training = is_training, decay=0.9, epsilon=1e-3,center=True,scale=True,updates_collections=None)
        print(1)
    if activation == 'relu':
        x = tf.nn.relu(x)
    elif activation == 'elu':
        x = tf.nn.elu(x)
    if pool:
        x = tf.layers.max_pooling3d(x, psize, strides=pstrides, padding=padding)
    if l2_loss:
        l2_losss = l2_losss + tf.nn.l2_loss(weight[varname_conv])

    return x, weight, l2_losss
conv3d.counter = 0
## dense layer module
def dense(x, weight, output_size, activation='none', l2_losss = 0, l2_loss=False,batch_norm = False, is_training = False,init_counter = False):
    if init_counter:
        dense.counter = 0
    dense.counter += 1
    varname_conv = 'dense_conv' + str(dense.counter)
    varname_bias = 'dense_bias' + str(dense.counter)

    input_size = np.prod(x.get_shape().as_list()[1:])

    weight[varname_conv] = tf.get_variable(shape=[input_size, output_size], name=varname_conv, initializer=tf.contrib.layers.xavier_initializer(uniform=False))
    weight[varname_bias] = tf.Variable(tf.zeros(output_size), name= varname_bias)
    x = tf.reshape(x, [-1] + [input_size])
    x = tf.matmul(x, weight[varname_conv])
    if batch_norm:
        x = tf.contrib.layers.batch_norm(x, is_training = is_training, decay=0.9, epsilon=1e-3,center=True,scale=True,updates_collections=None)
    if activation == 'relu':
        x = tf.nn.relu(x)
    elif activation == 'elu':
        x = tf.nn.elu(x)
    if l2_loss:
        l2_losss = l2_losss + tf.nn.l2_loss(weight[varname_conv])
    return x, weight, l2_losss
dense.counter = 0
########################################################################################################################
## Graph input
with tf.name_scope("input"):
    data = tf.placeholder(dtype=tf.float32, shape=[None] + input_shape, name='data')
    dataIdx = tf.placeholder(dtype=tf.float32, shape=[None, n_classes], name='dataIdx')
    lr = tf.placeholder(dtype=tf.float32, name='lr')
    is_training = tf.placeholder(dtype=tf.bool, name='is_training')
    weight = {}
    l2_loss = 0
########################################################################################################################
## Create model
# 3DCNN model
with tf.variable_scope("3DCNN"):
    data_reshape_Pow = tf.expand_dims(data, -1)
    [LayerPow_s1, weight, l2_loss] = conv3d(data_reshape_Pow, weight, [3, 3, 3, 1, 32], kstrides=[1, 1, 1], activation='elu', pool=False, psize=[2, 2, 1], pstrides=[2, 2, 1], padding='SAME',
                                            l2_losss=l2_loss, l2_loss=True, batch_norm=False, is_training=is_training)
    LayerPow_f1_ = tf.layers.max_pooling3d(LayerPow_s1, [2, 2, 2], strides=[2, 2, 2], padding='SAME')

    [LayerPow_s2, weight, l2_loss] = conv3d(LayerPow_f1_, weight, [3, 3, 3, 32, 32], kstrides=[1, 1, 1], activation='elu', pool=False, psize=[2, 2, 1], pstrides=[2, 2, 1], padding='SAME',
                                            l2_losss=l2_loss, l2_loss=True, batch_norm=False, is_training=is_training)
    LayerPow_f2_ = tf.layers.max_pooling3d(LayerPow_s2, [2, 2, 2], strides=[2, 2, 2], padding='SAME')

    [LayerPow_s3, weight, l2_loss] = conv3d(LayerPow_f2_, weight, [3, 3, 3, 32, 64], kstrides=[1, 1, 1], activation='elu', pool=False, psize=[2, 2, 1], pstrides=[2, 2, 1], padding='SAME',
                                            l2_losss=l2_loss, l2_loss=True, batch_norm=False, is_training=is_training)
    LayerPow_f3_ = tf.layers.max_pooling3d(LayerPow_s3, [2, 2, 2], strides=[2, 2, 2], padding='SAME')

    [LayerPow_s4, weight, l2_loss] = conv3d(LayerPow_f3_, weight, [3, 3, 3, 64, 64], kstrides=[1, 1, 1], activation='elu', pool=False, psize=[2, 2, 1], pstrides=[2, 2, 1], padding='SAME',
                                            l2_losss=l2_loss, l2_loss=True, batch_norm=False, is_training=is_training)
    LayerPow_f4_ = tf.layers.max_pooling3d(LayerPow_s4, [2, 2, 2], strides=[2, 2, 2], padding='SAME')

# Multilevel fusion
with tf.variable_scope("Fusion"):
    # extracting 1-dimensional multilevel feature
    LayerPow_f1__pooled = tf.math.reduce_mean((LayerPow_f1_), axis=4, keepdims=True)
    LayerPow_f2__pooled = tf.math.reduce_mean((LayerPow_f2_), axis=4, keepdims=True)
    LayerPow_f3__pooled = tf.math.reduce_mean((LayerPow_f3_), axis=4, keepdims=True)

    LayerPow_f1__pooled1 = tf.math.reduce_max((LayerPow_f1_), axis=4, keepdims=True)
    LayerPow_f2__pooled1 = tf.math.reduce_max((LayerPow_f2_), axis=4, keepdims=True)
    LayerPow_f3__pooled1 = tf.math.reduce_max((LayerPow_f3_), axis=4, keepdims=True)
    LayerPow_f1__pooled = tf.concat([LayerPow_f1__pooled, LayerPow_f1__pooled1], axis=4)
    LayerPow_f2__pooled = tf.concat([LayerPow_f2__pooled, LayerPow_f2__pooled1], axis=4)
    LayerPow_f3__pooled = tf.concat([LayerPow_f3__pooled, LayerPow_f3__pooled1], axis=4)

    LayerPow_f1__pooled = tf.contrib.layers.dropout(LayerPow_f1__pooled, keep_prob=0.5, is_training=is_training)
    LayerPow_f2__pooled = tf.contrib.layers.dropout(LayerPow_f2__pooled, keep_prob=0.5, is_training=is_training)
    LayerPow_f3__pooled = tf.contrib.layers.dropout(LayerPow_f3__pooled, keep_prob=0.5, is_training=is_training)
    LayerPow_f4_ = tf.contrib.layers.dropout(LayerPow_f4_, keep_prob=0.5, is_training=is_training)
    [Fc1, weight, l2_loss] = dense((LayerPow_f1__pooled), weight, 128, activation='elu', l2_losss=l2_loss, l2_loss=True, is_training=is_training)
    [Fc2, weight, l2_loss] = dense((LayerPow_f2__pooled), weight, 128, activation='elu', l2_losss=l2_loss, l2_loss=True, is_training=is_training)
    [Fc3, weight, l2_loss] = dense((LayerPow_f3__pooled), weight, 128, activation='elu', l2_losss=l2_loss, l2_loss=True, is_training=is_training)
    [Fc4, weight, l2_loss] = dense(LayerPow_f4_, weight, 128, activation='elu', l2_losss=l2_loss, l2_loss=True, is_training=is_training)
    Fc1_ = tf.nn.l2_normalize(Fc1,axis=1)
    Fc2_ = tf.nn.l2_normalize(Fc2,axis=1)
    Fc3_ = tf.nn.l2_normalize(Fc3,axis=1)
    Fc4_ = tf.nn.l2_normalize(Fc4,axis=1)
    # extracting weighting factors
    layer_attention = tf.nn.l2_normalize(tf.concat([Fc1, Fc2, Fc3, Fc4],axis=1), axis=1)
    [layer_attention, weight, l2_loss] = dense((layer_attention), weight, 4, activation='none', l2_losss=l2_loss, l2_loss=True, batch_norm=False, is_training=is_training)
    layer_attention1 = tf.nn.softmax(layer_attention,axis=1)
    # gradient compensation
    Fc1_ = (1 - tf.stop_gradient(tf.math.reciprocal(tf.expand_dims(layer_attention1[:, 0], -1)))) * tf.stop_gradient(Fc1_) + tf.stop_gradient(
        tf.math.reciprocal(tf.expand_dims(layer_attention1[:, 0], -1))) * Fc1_
    Fc2_ = (1 - tf.stop_gradient(tf.math.reciprocal(tf.expand_dims(layer_attention1[:, 1], -1)))) * tf.stop_gradient(Fc2_) + tf.stop_gradient(
        tf.math.reciprocal(tf.expand_dims(layer_attention1[:, 1], -1))) * Fc2_
    Fc3_ = (1 - tf.stop_gradient(tf.math.reciprocal(tf.expand_dims(layer_attention1[:, 2], -1)))) * tf.stop_gradient(Fc3_) + tf.stop_gradient(
        tf.math.reciprocal(tf.expand_dims(layer_attention1[:, 2], -1))) * Fc3_
    Fc4_ = (1 - tf.stop_gradient(tf.math.reciprocal(tf.expand_dims(layer_attention1[:, 3], -1)))) * tf.stop_gradient(Fc4_) + tf.stop_gradient(
        tf.math.reciprocal(tf.expand_dims(layer_attention1[:, 3], -1))) * Fc4_
    # classification layer
    weight['classifier'] = tf.get_variable(shape=[128, 2], name='classifier', initializer=tf.contrib.layers.xavier_initializer(uniform=False))
    Fc11__ = tf.matmul(Fc1_, tf.nn.l2_normalize(weight['classifier'],axis=0))
    Fc22__ = tf.matmul(Fc2_, tf.nn.l2_normalize(weight['classifier'],axis=0))
    Fc33__ = tf.matmul(Fc3_, tf.nn.l2_normalize(weight['classifier'],axis=0))
    Fc44__ = tf.matmul(Fc4_, tf.nn.l2_normalize(weight['classifier'],axis=0))

    Fc11__ = Fc11__ * tf.expand_dims(layer_attention1[:, 0], -1)
    Fc22__ = Fc22__ * tf.expand_dims(layer_attention1[:, 1], -1)
    Fc33__ = Fc33__ * tf.expand_dims(layer_attention1[:, 2], -1)
    Fc44__ = Fc44__ * tf.expand_dims(layer_attention1[:, 3], -1)

    PredFusion = Fc11__ + Fc22__ + Fc33__ + Fc44__
    # loss rescaling
    weight['rescaling'] = tf.Variable(np.array([1],dtype=np.float32))
    PredFusion = PredFusion*tf.math.square(weight['rescaling'])
########################################################################################################################
# Define loss and optimizer
with tf.variable_scope("cost"):
    costFusion = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=PredFusion,labels=dataIdx))
    optimizer_fusion = tf.train.AdamOptimizer(learning_rate=lr).minimize(costFusion)
with tf.variable_scope("eval"):
    correct_pred_fusion = tf.equal(tf.argmax(PredFusion, 1), tf.argmax(dataIdx, 1))
    accuracy_fusion = tf.reduce_mean(tf.cast(correct_pred_fusion, tf.float32))
########################################################################################################################
########################################################################################################################
## Model training
print('Model training.........')
saver = tf.train.Saver()
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    epoch = 1
    while (epoch <= training_epoch):
        # Data shuffling
        sample_idx_random = random.sample(list(range(n_samples)), n_samples)
        sample_idx_random = np.array(sample_idx_random)
        # Decaying learning rate
        if (epoch == decaying_epoch):
            learning_rate = 0.1 * learning_rate
            print("***** Learning rate is decayed to {:.10f} *****".format(learning_rate))
        step = 0
        while step < n_samples // batch_size:
            # Batch selection and train model
            sample_idx = sample_idx_random[step * batch_size: (step + 1) * batch_size]
            data_batch = data_train[sample_idx, :]
            dataIdx_batch = dataIdx_train[sample_idx, :]
            sess.run(optimizer_fusion, feed_dict={data: data_batch, dataIdx: dataIdx_batch, lr: learning_rate, is_training: True})
            step += 1
        # Display training accuracy and loss
        if (epoch % display_epoch_train == 0) or (epoch < 10) or (epoch > training_epoch - 6):
            [loss, acc_train] = sess.run([costFusion, accuracy_fusion], feed_dict={data: data_batch, dataIdx: dataIdx_batch, is_training: False})
            end_time = time.time()
            now = time.localtime()
            print("epoch {0}".format(epoch) + "/{0} : ".format(training_epoch) +
                  "train acc = {:.6f},  ".format(acc_train) + "loss = {:.5f},  ".format(loss) +
                  "running time = {:.1f} min ".format((end_time - start_time) / 60) +
                  "(time : %02d:%02d:%02d)" % (now.tm_hour, now.tm_min, now.tm_sec))
        # Saving model and displaying test accuracy
        if (epoch % display_epoch_test == 0) or (epoch == training_epoch) or (epoch > training_epoch - 6):
            acc_test = 0
            for i in range(dataIdx_test.shape[0]):
                data_batch = data_test[i, :]
                data_batch = np.expand_dims(data_batch, axis=0)
                dataIdx_batch = dataIdx_test[i, :]
                dataIdx_batch = np.expand_dims(dataIdx_batch, axis=0)
                acc_test += sess.run(accuracy_fusion, feed_dict={data: data_batch, dataIdx: dataIdx_batch, is_training: False})
            acc_test = float(acc_test) / dataIdx_test.shape[0]
            print("Test Accuracy = " + "{:.5f}".format(acc_test))
            save_path1 = saver.save(sess, save_path+model_name +"_epoch"+str(epoch))
            print("Save model at '" +save_path+model_name +"_epoch"+str(epoch))
        epoch += 1
    sess.close()
