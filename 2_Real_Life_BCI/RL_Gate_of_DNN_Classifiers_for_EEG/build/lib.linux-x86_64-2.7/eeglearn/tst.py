from __future__ import print_function
import time

import numpy as np
np.random.seed(1234)
from functools import reduce
import math as m

import scipy.io
import theano
import theano.tensor as T

from scipy.interpolate import griddata
from sklearn.preprocessing import scale
from utils import cart2sph, pol2cart, augment_EEG

import lasagne
#from lasagne.layers.dnn import Conv2DDNNlayer as ConvLayer
from lasagne.layers import Conv2DLayer, MaxPool2DLayer, InputLayer
from lasagne.layers import DenseLayer, ElemwiseMergeLayer, FlattenLayer
from lasagne.layers import ConcatLayer, ReshapeLayer, get_output_shape
from lasagne.layers import Conv1DLayer, DimshuffleLayer, LSTMLayer, SliceLayer

import tensorflow as tf
import cv2
from matplotlib import pyplot as plt
import os

print(os.getcwd())


# define initial weight & variable for tf
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# define
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    print((x.shape, W.shape))

    print(x)
    print(W)
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return x
    #return tf.nn.relu(x)

def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')

def azim_proj(pos):
    """
    computes the azimuthal equidistant projection of input point in 3d cartesian coordinates.
    imagine a plane being placed against (tangent to) a globe. if
    a light source inside the globe projects the graticule onto
    the plane the result would be a planar, or azimuthal, map
    projection.

    :param pos: position in 3d cartesian coordinates
    :return: projected coordinates using azimuthal equidistant projection
    """
    [r, elev, az] = cart2sph(pos[0], pos[1], pos[2])
    return pol2cart(az, m.pi / 2 - elev)


def gen_images(locs, features, n_gridpoints, normalize=True,
               augment=False, pca=False, std_mult=0.1, n_components=2, edgeless=False):
    """
    generates eeg images given electrode locations in 2d space and multiple feature values for each electrode

    :param locs: an array with shape [n_electrodes, 2] containing x, y
                        coordinates for each electrode.
    :param features: feature matrix as [n_samples, n_features]
                                features are as columns.
                                features corresponding to each frequency band are concatenated.
                                (alpha1, alpha2, ..., beta1, beta2,...)
    :param n_gridpoints: number of pixels in the output images
    :param normalize:   flag for whether to normalize each band over all samples
    :param augment:     flag for generating augmented images
    :param pca:         flag for pca based data augmentation
    :param std_mult     multiplier for std of added noise
    :param n_components: number of components in pca to retain for augmentation
    :param edgeless:    if true generates edgeless images by adding artificial channels
                        at four corners of the image with value = 0 (default=false).
    :return:            tensor of size [samples, colors, w, h] containing generated
                        images.
    """
    feat_array_temp = []
    nelectrodes = locs.shape[0]     # number of electrodes
    # test whether the feature vector length is divisible by number of electrodes
    assert features.shape[1] % nelectrodes == 0
    n_colors = features.shape[1] / nelectrodes
    for c in range(n_colors):
        feat_array_temp.append(features[:, c * nelectrodes : nelectrodes * (c+1)])
    if augment:
        if pca:
            for c in range(n_colors):
                feat_array_temp[c] = augment_eeg(feat_array_temp[c], std_mult, pca=true, n_components=n_components)
        else:
            for c in range(n_colors):
                feat_array_temp[c] = augment_eeg(feat_array_temp[c], std_mult, pca=false, n_components=n_components)
    nsamples = features.shape[0]
    # interpolate the values
    grid_x, grid_y = np.mgrid[
                     min(locs[:, 0]):max(locs[:, 0]):n_gridpoints*1j,
                     min(locs[:, 1]):max(locs[:, 1]):n_gridpoints*1j
                     ]
    temp_interp = []
    for c in range(n_colors):
        temp_interp.append(np.zeros([nsamples, n_gridpoints, n_gridpoints]))
    # generate edgeless images
    if edgeless:
        min_x, min_y = np.min(locs, axis=0)
        max_x, max_y = np.max(locs, axis=0)
        locs = np.append(locs, np.array([[min_x, min_y], [min_x, max_y],[max_x, min_y],[max_x, max_y]]),axis=0)
        for c in range(n_colors):
            feat_array_temp[c] = np.append(feat_array_temp[c], np.zeros((nsamples, 4)), axis=1)
    # interpolating
    for i in xrange(nsamples):
        for c in range(n_colors):
            temp_interp[c][i, :, :] = griddata(locs, feat_array_temp[c][i, :], (grid_x, grid_y),
                                    method='cubic', fill_value=np.nan)
        print('interpolating {0}/{1}\r'.format(i+1, nsamples), end='\r')
    # normalizing
    for c in range(n_colors):
        if normalize:
            temp_interp[c][~np.isnan(temp_interp[c])] = \
                scale(temp_interp[c][~np.isnan(temp_interp[c])])
        temp_interp[c] = np.nan_to_num(temp_interp[c])
    return np.swapaxes(np.asarray(temp_interp), 0, 1)     # swap axes to have [samples, colors, w, h]


#create model cnn
def build_cnn(input_var=None, w_init=None, n_layers=(4, 2, 1), n_filters_first=32, imsize=32, n_colors=3, filtersize=(3, 3)):
    """  """
    weights = []        # Keeps the weights for all layers
    bias = []
    count = 0
    n_classes = 4
    #x = tf.placeholder("float", shape=[None, imsize*imsize, n_colors])
    if input_var is None:
        x = tf.placeholder("float", shape=[None, imsize*imsize, n_colors])
    else:
        x = input_var
    y = tf.placeholder("float", shape=[None, n_classes])
    x_image = tf.reshape(x, [-1, imsize, imsize, n_colors])

    inkernel = 3
    outkernel = n_filters_first
    if w_init is None:
        for i, s in enumerate(n_layers):
            w_tmp = []
            b_tmp = []
            h_tmp = []
            for l in range(s):
                w_tmp.append(weight_variable([filtersize[0], filtersize[1], inkernel, outkernel]))
                b_tmp.append(bias_variable([outkernel]))
                print('helpppp:', l)
                print(w_tmp[l].shape)
                inkernel = outkernel
#            inkernel = outkernel
            outkernel = outkernel*2
            weights.append(w_tmp)
            bias.append(b_tmp)

        #fully connected layer's weight
        weights.append(weight_variable([outkernel, 512])) 
        weights.append(weight_variable([512, n_classes]))
        bias.append(bias_variable([512]))
        bias.append(bias_variable([n_classes]))

    h_conv = []  # system function
    h_pool = []
    x_datain = x_image
    for i, s in enumerate(n_layers):
        h_conv_tmp = []
        for l in range(s):
            #define h
            print('i,l')
            print((i, l))
            h_convout = tf.nn.relu(conv2d(x_datain, weights[i][l], bias[i][l]))
            h_conv_tmp.append(h_convout)
            print(x_datain.shape ,h_convout.shape)

            x_datain = h_convout
        h_conv.append(h_conv_tmp) #save result
        #max pooling layer
        h_convout = maxpool2d(x_datain, 2)
        h_pool.append(h_convout) #save reuslt
        x_datain = h_convout
    return h_convout, weights


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    """
    Iterates over the samples returing batches of size batchsize.
    :param inputs: input data array. It should be a 4D numpy array for images [n_samples, n_colors, W, H] and 5D numpy
                    array if working with sequence of images [n_timewindows, n_samples, n_colors, W, H].
    :param targets: vector of target labels.
    :param batchsize: Batch size
    :param shuffle: Flag whether to shuffle the samples before iterating or not.
    :return: images and labels for a batch
    """
    if inputs.ndim == 4:
        input_len = inputs.shape[0]
    elif inputs.ndim == 5:
        input_len = inputs.shape[1]
    assert input_len == len(targets)
    if shuffle:
        indices = np.arange(input_len)
        np.random.shuffle(indices)
    for start_idx in range(0, input_len, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        if inputs.ndim == 4:
            yield inputs[excerpt], targets[excerpt]
        elif inputs.ndim == 5:
            yield inputs[:, excerpt], targets[excerpt]




from utils import reformatInput

print('Loading data...')
locs = scipy.io.loadmat('../Sample data/Neuroscan_locs_orig.mat')
locs_3d = locs['A']
locs_2d = []

    # Convert to 2D
for e in locs_3d:
    locs_2d.append(azim_proj(e))


feats = scipy.io.loadmat('../Sample data/FeatureMat_timeWin.mat')['features']
subj_nums = np.squeeze(scipy.io.loadmat('../Sample data/trials_subNums.mat')['subjectNum'])
    # Leave-Subject-Out cross validation
fold_pairs = []
for i in np.unique(subj_nums):
    ts = subj_nums == i
    tr = np.squeeze(np.nonzero(np.bitwise_not(ts)))
    ts = np.squeeze(np.nonzero(ts))
    np.random.shuffle(tr)  # Shuffle indices
    np.random.shuffle(ts)
    fold_pairs.append((tr, ts))

    # CNN Mode
print('Generating images...')
    # Find the average response over time windows
av_feats = reduce(lambda x, y: x+y, [feats[:, i*192:(i+1)*192] for i in range(feats.shape[1] / 192)])
av_feats = av_feats / (feats.shape[1] / 192)
images = gen_images(np.array(locs_2d), av_feats, 32, normalize=False)
print('\n')
print(images)
print(images.shape)

    # Class labels should start from 0
print('Training the CNN Model...')
train(images, np.squeeze(feats[:, -1]) - 1, fold_pairs[2], 'cnn')

    # Construct model
pred = conv_net(x, weights, biases, keep_prob)

    # Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))




    print("start training")
    # Initializing the variables
    init = tf.global_variables_initializer()
    display_step = 1
    xin = input_var
    yout = output_var

    with tf.Session() as sess:
        writer = tf.summary.FileWriter('logs', sess.graph)
        sess.run(init)
        best_val_acc = 0
        for epoch in range(num_epochs):
            train_loss = 0
            val_loss = 0
            train_batches = 0
            start_time = time.time()
            train_step = 1
            val_step = 1
            val_batches = 0
            sum_train_acc = 0
            sum_train_loss = 0
            sum_val_acc = 0
            sum_val_loss = 0

            # train accuracy
            for batch in iterate_minibatches(X_train, y_train, batch_size, shuffle=False):
                inputs, targets = batch
                targets = np.eye(num_classes)[targets.reshape(-1)]
                '''
                #check for monitor size
                if train_step ==1:
                    print("batch info is.. batch, inputs.shape, targets.shape")
                    print(X_train.shape)
                    print(y_train.shape)
                    print("batch info")
                    print(inputs.shape)
                    print(targets.shape)
                    print(xin.shape)
                    print(yout.shape)
                '''
                sess.run(optimizer, feed_dict = {xin: inputs, yout: targets, keep_prob: 0.5})
                
                if train_step % display_step == 0:
                    #calculate batch loss & accuracy
                    train_loss, train_acc = sess.run([cost, accuracy], feed_dict = {xin: inputs, yout: targets, keep_prob: 1.0})
                    print("Iter " + str(train_step*batch_size) + ", Minibatch Loss= " + \
                            "{:.6f}".format(train_loss) + ", Training Accuracy= " + \
                            "{:.5f}".format(train_acc))
                    
                    sum_train_loss += train_loss
                    sum_train_acc += train_acc
                    print(train_loss, sum_train_loss, train_acc, sum_train_acc)
                train_step += 1

            print("train Opt fin")
            # validation accuracy
            for batch in iterate_minibatches(X_val, y_val, batch_size, shuffle=False):
                inputs, targets = batch
                targets = np.eye(num_classes)[targets.reshape(-1)]
                sess.run(optimizer, feed_dict = {xin: inputs, yout: targets, keep_prob: 0.5})
                
                if val_step % display_step == 0:
                    #calculate batch loss & accuracy
                    val_acc = sess.run(accuracy, feed_dict = {xin: inputs, yout: targets, keep_prob: 1.0})
                    print("Iter " + str(val_step*batch_size) + ", Minibatch Loss= " + \
                            ", cross validatino  Accuracy= " + "{:.5f}".format(val_acc))

                    #sum_val_loss += val_loss
                    sum_val_acc += val_acc

                    print(val_acc, sum_val_acc)
                val_step += 1

            print("Validation Opt fin")
            
            #calculate average loss & accuracy
            av_train_loss = sum_train_loss/train_step*display_step
            av_train_acc = sum_train_acc/train_step*display_step
            #av_val_loss = sum_val_loss/val_step*display_step
            av_val_acc = sum_val_acc/val_step*display_step

            print(av_train_loss, av_train_acc, av_val_acc)

            # Then we print the results for this epoch:
            print("Epoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, time.time() - start_time))
            print("  training loss:\t\t{:.6f}".format(av_train_loss))
            #print("  validation loss:\t\t{:.6f}".format(av_val_loss))
            print("  validation accuracy:\t\t{:.2f} %".format(av_val_acc * 100))

            if av_val_acc > best_val_acc:
                best_val_acc = av_val_acc
                # After training, we compute & print the test error
                sum_test_err = 0
                sum_test_acc = 0
                test_batches = 0
                
                for batch in iterate_minibatches(X_test, y_test, batch_size, shuffle=False):
                    inputs, targets = batch
                    targets = np.eye(num_classes)[targets.reshape(-1)]

                    test_acc = sess.run(accuracy, feed_dict={xin: inputs, yout: targets, keep_prob: 1.0})
                    print(test_acc)
                    sum_test_acc += test_acc
                    test_batches += 1

                av_test_acc = sum_test_acc / test_batches
                print("Final results:")
                #print("  test loss:\t\t\t{:.6f}".format(av_test_err))
                print("  test accuracy:\t\t{:.2f} %".format(av_test_acc * 100))
                
                # Dump the network weights to a file like this:
                #tf.saver.save(sess, 'my-model')
                # `save` method will call `export_meta_graph` implicitly.
                # you will get saved graph files:my-model.meta
        print('-'*50)
        print("Best validation accuracy:\t\t{:.2f} %".format(best_val_acc * 100))
        print("Best test accuracy:\t\t{:.2f} %".format(av_test_acc * 100))
        writer = tf.summary.FileWriter('logs', sess.graph)
